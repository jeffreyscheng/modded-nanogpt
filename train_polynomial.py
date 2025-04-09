import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import pickle
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import math
import uuid
import time
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
from torch import Tensor

# Function to round mantissa bits to simulate specific precision
def round_to_precision(param, bits):
    """
    Properly simulates reduced precision floating point by rounding mantissa bits.
    
    Float32 has: 1 sign bit + 8 exponent bits + 23 mantissa bits
    BFloat16 has: 1 sign bit + 8 exponent bits + 7 mantissa bits
    
    So we need to round 16 bits going from float32 to bfloat16.
    
    Args:
        param: A float32 tensor
        bits: Target bits of precision (32 to 16)
    
    Returns:
        A float32 tensor with reduced precision
    """
    if bits >= 32:
        return param  # No reduction needed
    
    # Calculate how many mantissa bits to keep (23 for float32, less for lower precision)
    mantissa_bits_to_keep = max(bits - 9, 0)  # 9 bits for sign+exponent
    bits_to_round = 23 - mantissa_bits_to_keep
    
    if bits_to_round <= 0:
        return param
    
    # Create a rounding bit at the position we're cutting off
    rounding_bit = 1 << (bits_to_round - 1)
    # Create a mask for the bits we want to keep
    mask = (-1) << bits_to_round
    
    # Cast to int to perform bit manipulation
    as_int = param.view(torch.int32)
    
    # Add rounding bit (adds 0.5 to the last bit being kept)
    rounded_int = as_int + rounding_bit
    
    # Apply mask to clear lower bits
    rounded_int = rounded_int & mask
    
    return rounded_int.view(torch.float32)

# Hook to maintain quantization level during gradient updates
class RoundingHook:
    def __init__(self, bits):
        self.bits = bits
        
    def __call__(self, grad):
        if grad is None:
            return None
        return round_to_precision(grad, self.bits)

@torch.compile
def zeropower_via_newtonschulz5(G, coefficients) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    X = G.bfloat16()  # Use bfloat16 for better performance
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for coeff in coefficients:
        a, b, c = coeff[0], coeff[1], coeff[2]  # Direct indexing to maintain gradients
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class GeneralizedNewtonSchulz(nn.Module):
    def __init__(self, init_coefficients: List[float] = None):
        super().__init__()
        # Create parameters with explicit dtype to ensure they require gradients
        self.poly_layers = nn.ParameterList([
            nn.Parameter(torch.tensor(coeff_tuple, dtype=torch.float32)) 
            for coeff_tuple in init_coefficients
        ])
        
    @property
    def degree(self) -> int:
        return len(self.poly_layers[0])

    @property
    def num_iterations(self) -> int:
        return len(self.poly_layers)
    
    @property
    def num_polynomial_terms(self) -> int:
        return (self.degree + 1) // 2

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Process batches efficiently 
        if X.ndim == 3:  # Handle batched input [batch, m, n]
            return self._forward_batch(X)
        else:  # Handle single matrix input [m, n]
            return zeropower_via_newtonschulz5(X, self.poly_layers)
    
    def _forward_batch(self, X: torch.Tensor) -> torch.Tensor:
        # Efficiently process batch of matrices
        batch_size = X.size(0)
        
        # Try to process in one go if batch size is reasonable
        if batch_size <= 32:  # Threshold based on profiling
            return zeropower_via_newtonschulz5(X, self.poly_layers)
        
        # Process larger batches in chunks to avoid OOM
        chunk_size = 16
        results = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk = X[i:end_idx]
            results.append(zeropower_via_newtonschulz5(chunk, self.poly_layers))
            
        return torch.cat(results, dim=0)
    
    def derivative_at_zero(self) -> torch.Tensor:
        # return the product of the zeroth coefficient in each layer of self.poly_layers
        # must be differentiable
        coeffs = torch.stack([layer[0] for layer in self.poly_layers])
        return torch.prod(coeffs)

    def print_polynomial(self) -> str:
        list_terms = ''
        for i, poly_coeffs in enumerate(self.poly_layers):
            list_terms += f"{poly_coeffs.tolist()},\n"
        desmos_terms = ''
        for i, poly_coeffs in enumerate(self.poly_layers):
            desmos_terms += f"f_{i}(x) = {poly_coeffs[0]}x + {poly_coeffs[1]}x^3 + {poly_coeffs[2]}x^5\n"
        return list_terms + desmos_terms

    def evaluate_scalar(self, x: torch.Tensor) -> torch.Tensor:
        for poly_coeffs in self.poly_layers:
            powers = torch.tensor([2 * i + 1 for i in range(len(poly_coeffs))])
            x = sum(coeff * torch.pow(x, power) for coeff, power in zip(poly_coeffs, powers))
        return x

def univariate_loss(model, device, lower_bound: float = 0.0, upper_bound: float = 1.0):
    mesh_size = 1_000_000
    x = torch.rand(mesh_size).to(device) * (upper_bound - lower_bound) + lower_bound
    y = model.evaluate_scalar(x)
    return torch.norm(y - 1, p='fro') / mesh_size

def train_newton_schulz(config: Dict[str, Any]):
    # Initialize distributed training
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    dist.barrier()
    master_process = (rank == 0)  # this process will do logging, checkpointing etc.
    
    # Configure wandb only on master process
    if master_process:
        run = wandb.init(project="newton-schulz-polynomial", config=config)
        # Begin logging
        run_id = uuid.uuid4()
        os.makedirs("polynomial_logs", exist_ok=True)
        logfile = f"polynomial_logs/{run_id}.txt"
        print(logfile)
        
        # begin by printing this file (the Python code)
        with open(logfile, "a") as f:
            print(code, file=f)
            print("="*100, file=f)
            print(f"Running Python {sys.version}", file=f)
            print(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}", file=f)
            print("="*100, file=f)
    else:
        run = None
        run_id = None
    
    # Helper function to log to master process
    def log(s, console=False):
        if master_process:
            with open(logfile, "a") as f:
                if console:
                    print(s)
                print(s, file=f)
    
    # Convert init_coefficients to ensure they're proper Python floats
    init_coefficients = [
        tuple(map(float, coeff_tuple)) for coeff_tuple in config['init_coefficients']
    ]
    
    model = GeneralizedNewtonSchulz(
        init_coefficients=init_coefficients
    ).to(device)
    
    # Broadcast model parameters from rank 0 to all processes
    for param in model.parameters():
        dist.broadcast(param.data, 0)
    
    # Log the learning rates if master process
    if master_process:
        log(f"Base learning rate: {config['learning_rate']}", console=True)
        log(f"Last two polynomial learning rate: {config['learning_rate'] * 5.0}", console=True)
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=config['adam_betas']
    )
    
    def plot_fn(fn):
        if not master_process:
            return None
            
        fig, ax = plt.subplots()
        x = torch.linspace(-0.1, 1.1, 100000).to(device)
        y = fn(x).cpu().detach()
        ax.plot(x.cpu().numpy(), y.numpy())
        ax.set_xlim(-0.11, 1.2)
        ax.set_ylim(-1.0, 2.0)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        img = wandb.Image(plt)
        plt.close()
        return img
    
    # Start timing
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    total_train_time = 0
    
    # Performance configuration
    torch.set_float32_matmul_precision('high')
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = config['learning_rate']
    
    # Gradient clipping threshold
    grad_clip_threshold = config.get('max_grad_norm', 0.5)
    
    # Compile the model (optional)
    if config.get('use_compile', True):
        model = torch.compile(model, dynamic=False, fullgraph=True)
    
    # Initialize precision bits (start with float32)
    start_bits = 32
    target_bits = 16
    current_bits = start_bits
    
    # Setup for plateau detection
    loss_history = []
    univariate_loss_history = []  # Track univariate loss separately for plateau detection
    min_epochs_before_reduction = config.get('min_epochs_before_reduction', 50)
    plateau_patience = config.get('plateau_patience', 20)
    plateau_threshold = config.get('plateau_threshold', 0.001)
    
    # Dictionary to store hooks
    hooks = {}
    
    # Apply initial hooks
    for param in model.parameters():
        hook = param.register_hook(RoundingHook(current_bits))
        hooks[id(param)] = hook
    
    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        num_batches = 0
        current_loss = 0
        epoch_univariate_loss = 0  # Track univariate loss for the epoch
        epoch_start_time = time.perf_counter()

        # Add a small delay between epochs to ensure all processes are synchronized
        if epoch > 0:
            torch.cuda.synchronize()
            dist.barrier()

        for i in range(1000):
            current_elapsed_time = time.perf_counter() - start_time
            univariate_loss_value = univariate_loss(model, device)
            derivative_reward_value = model.derivative_at_zero()
            loss = config['univariate_loss_weight'] * univariate_loss_value - config['derivative_reward_weight'] * derivative_reward_value
            
            if master_process and i % 10 == 0:
                log(f"Epoch {epoch}, batch {i}: univariate loss: {univariate_loss_value.item():.4e}, "
                    f"derivative: {derivative_reward_value.item():.4e}, Elapsed: {current_elapsed_time:.2f}s", console=True)
            
            loss.backward()
            
            # Synchronize gradients across all processes
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            
            # Enable gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_threshold)
            optimizer.step()
            
            # Ensure parameters maintain truncation after optimizer update
            with torch.no_grad():
                for param in model.parameters():
                    param.copy_(round_to_precision(param, current_bits))
            
            optimizer.zero_grad(set_to_none=True)
            
            current_loss += loss.item()
            epoch_loss += loss.item()
            epoch_univariate_loss += univariate_loss_value.item()  # Accumulate univariate loss
            num_batches += 1
            
            # Log to wandb periodically
            if (i + 1) % config['accumulation_steps'] == 0 and master_process:
                wandb.log({
                    "loss": current_loss / config['accumulation_steps'],
                    "derivative": derivative_reward_value.item(),
                    "univariate_loss": univariate_loss_value.item(),
                    "current_bits": current_bits,
                    "mantissa_bits": current_bits - 9
                })
                current_loss = 0
            
            # Free memory
            torch.cuda.empty_cache() if config.get('aggressive_memory_cleanup', False) else None

        # Log epoch statistics
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start_time
        total_train_time += epoch_time
        
        # Calculate average loss for this epoch
        avg_epoch_loss = epoch_loss/max(1, num_batches)
        avg_univariate_loss = epoch_univariate_loss/max(1, num_batches)  # Calculate average univariate loss
        loss_history.append(avg_epoch_loss)
        univariate_loss_history.append(avg_univariate_loss)  # Store for plateau detection
        
        # Check for plateau and reduce precision if needed - use univariate loss for plateau detection
        if epoch >= min_epochs_before_reduction and current_bits > target_bits:
            # Check last plateau_patience epochs for plateauing
            if len(univariate_loss_history) >= plateau_patience:
                recent_losses = univariate_loss_history[-plateau_patience:]
                
                # Calculate relative change in univariate loss
                start_loss = recent_losses[0]
                end_loss = recent_losses[-1]
                loss_change = abs(end_loss - start_loss) / (abs(start_loss) + 1e-10)
                
                # Check if univariate loss has plateaued
                if loss_change < plateau_threshold:
                    # Reduce precision by 1 bit
                    current_bits -= 1
                    current_bits = max(current_bits, target_bits)
                    
                    if master_process:
                        log(f"Univariate loss plateaued with change {loss_change:.6f} < threshold {plateau_threshold}. "
                            f"Reducing precision to {current_bits} bits (mantissa: {current_bits - 9} bits)", console=True)
                    
                    # Remove old hooks
                    for param_id in list(hooks.keys()):
                        hooks[param_id].remove()
                        del hooks[param_id]
                    
                    # Apply truncation and register new hooks
                    for param in model.parameters():
                        # Truncate the parameter
                        param.data.copy_(round_to_precision(param.data, current_bits))
                        
                        # Register hook for maintaining truncation during updates
                        hook = param.register_hook(RoundingHook(current_bits))
                        hooks[id(param)] = hook
                    
                    # Reset loss histories after reducing precision
                    loss_history = []
                    univariate_loss_history = []
        
        if master_process:
            log(f"Epoch {epoch}/{config['num_epochs']} completed in {epoch_time:.2f}s, avg_loss: {avg_epoch_loss:.6f}, "
                f"avg_univariate_loss: {avg_univariate_loss:.6f}, bits: {current_bits}", console=True)
            
            # Log parameter values in float32 and bfloat16 for comparison
            if epoch % 10 == 0:
                for i, param in enumerate(model.parameters()):
                    bfloat16_param = param.data.clone().bfloat16().float()
                    log(f"Parameter {i}, float32: {param.data.flatten()[0].item()}, bfloat16: {bfloat16_param.flatten()[0].item()}", console=True)
                    # Test if our truncation matches bfloat16
                    truncated = round_to_precision(param.data, 16)
                    log(f"Rounded to 16: {truncated.flatten()[0].item()}", console=True)
            
            wandb.log({
                "epoch": epoch,
                "avg_epoch_loss": avg_epoch_loss,
                "avg_univariate_loss": avg_univariate_loss,
                "epoch_time": epoch_time,
                "polynomial": plot_fn(model.evaluate_scalar),
                "current_bits": current_bits,
                "mantissa_bits": current_bits - 9
            })
            
            log(model.print_polynomial(), console=True)
    
    # Log final statistics
    torch.cuda.synchronize()
    if master_process:
        log(f"Training completed in {total_train_time:.2f}s", console=True)
        log(f"Final polynomial: {model.print_polynomial()}", console=True)
        log(f"Final derivative at zero: {model.derivative_at_zero().item()}", console=True)
        log(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
           f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
        
        # Convert final model to bfloat16 for verification
        bfloat16_model = model
        for param in bfloat16_model.parameters():
            param.data = param.data.bfloat16()
        
        # Test the bfloat16 model
        try:
            bfloat16_test = univariate_loss(bfloat16_model, device)
            log(f"bfloat16 univariate loss: {bfloat16_test.item():.4e}", console=True)
        except Exception as e:
            log(f"bfloat16 test failed: {str(e)}", console=True)
        
        # Finish wandb run
        if run is not None:
            run.finish()
    
    # Clean up
    for param_id in list(hooks.keys()):
        hooks[param_id].remove()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    default_config = {
        "device": "cuda",
        "checkpoint_dirs": ["matrices/a519aa14-cfde-4f80-aa58-87c9d9761a2a"],
        "learning_rate": 1e-5,
        "adam_betas": (0.8, 0.8),
        "batch_size": 64,  # Global batch size
        "batch_size_per_gpu": 8,  # Per-GPU batch size (will override global batch size)
        "accumulation_steps": 1,
        "max_grad_norm": 1e-3,
        "num_epochs": 5000,
        "min_epochs_before_reduction": 5,  # Minimum epochs before allowing precision reduction
        "plateau_patience": 5,  # Number of epochs to check for plateau
        "plateau_threshold": 0.001,  # Relative change threshold to detect plateau
        "derivative_reward_weight": 1e-5,
        "univariate_loss_weight": 1000,
        "save_checkpoints": True,
        "use_compile": True,  # Use torch.compile
        "aggressive_memory_cleanup": False,  # Enable for extreme OOM cases
        "init_coefficients": [
            [4.510918617248535, -7.218330383300781, 2.7313356399536133],
            [4.605247974395752, -6.573604106903076, 2.3541903495788574],
            [4.919572353363037, -5.963148593902588, 1.814536452293396],
            [4.132555961608887, -3.550161123275757, 0.7916041016578674],
            [3.9480252265930176, -3.3081469535827637, 0.7301044464111328],
            [2.495997905731201, -1.7606257200241089, 0.3820432126522064],
            [2.0578646659851074, -1.5518815517425537, 0.48931562900543213],
        ]
    }
    
    # Run single training
    train_newton_schulz(default_config)
    
    # Uncomment for hyperparameter sweep
    """
    sweep_config = {
        'method': 'grid',
        'parameters': {
            'degree': {'values': [3, 5, 7]},
            'num_iterations': {'values': [2, 3, 4, 5, 10]},
            'learning_rate': {'values': [0.001, 0.01, 0.1]},
        }
    }
    
    sweep_id = wandb.sweep(sweep_config, project="newton-schulz-polynomial")
    wandb.agent(sweep_id, function=train_newton_schulz)
    """