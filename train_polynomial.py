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
    def __init__(self, init_coefficients: List[float] = None, scaling_factor: float = 32.0):
        super().__init__()
        # Create parameters with explicit dtype to ensure they require gradients
        self.poly_layers = nn.ParameterList([
            nn.Parameter(torch.tensor(coeff_tuple, dtype=torch.bfloat16) * scaling_factor) 
            for coeff_tuple in init_coefficients
        ])
        
        # Store scaling factor as a buffer (not a parameter)
        self.register_buffer('scaling_factor', torch.tensor(scaling_factor, dtype=torch.bfloat16))
    
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
        # Get scaled coefficients
        scaled_poly_layers = [layer / self.scaling_factor for layer in self.poly_layers]
        
        # Process batches efficiently 
        if X.ndim == 3:  # Handle batched input [batch, m, n]
            return self._forward_batch(X, scaled_poly_layers)
        else:  # Handle single matrix input [m, n]
            return zeropower_via_newtonschulz5(X, scaled_poly_layers)
    
    def _forward_batch(self, X: torch.Tensor, scaled_poly_layers) -> torch.Tensor:
        # Efficiently process batch of matrices
        batch_size = X.size(0)
        
        # Try to process in one go if batch size is reasonable
        if batch_size <= 32:  # Threshold based on profiling
            return zeropower_via_newtonschulz5(X, scaled_poly_layers)
        
        # Process larger batches in chunks to avoid OOM
        chunk_size = 16
        results = []
        
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk = X[i:end_idx]
            results.append(zeropower_via_newtonschulz5(chunk, scaled_poly_layers))
            
        return torch.cat(results, dim=0)
    
    def derivative_at_zero(self) -> torch.Tensor:
        # Scale down the parameters before computing the derivative
        scaled_coeffs = torch.stack([layer[0] / self.scaling_factor for layer in self.poly_layers])
        return torch.prod(scaled_coeffs)

    def print_polynomial(self) -> str:
        list_terms = ''
        for i, poly_coeffs in enumerate(self.poly_layers):
            # Scale down for display
            scaled_coeffs = (poly_coeffs / self.scaling_factor).tolist()
            list_terms += f"{scaled_coeffs},\n"
        desmos_terms = ''
        for i, poly_coeffs in enumerate(self.poly_layers):
            # Scale down for display
            scaled_coeffs = poly_coeffs / self.scaling_factor
            desmos_terms += f"f_{i}(x) = {scaled_coeffs[0]}x + {scaled_coeffs[1]}x^3 + {scaled_coeffs[2]}x^5\n"
        return list_terms + desmos_terms

    def evaluate_scalar(self, x: torch.Tensor) -> torch.Tensor:
        for poly_coeffs in self.poly_layers:
            # Scale down the coefficients for computation
            scaled_coeffs = poly_coeffs / self.scaling_factor
            powers = torch.tensor([2 * i + 1 for i in range(len(scaled_coeffs))])
            x = sum(coeff * torch.pow(x, power) for coeff, power in zip(scaled_coeffs, powers))
        return x

@dataclass
class MatrixSample:
    name: str
    shape: Tuple[int, ...]
    file_path: str

class MatrixDataset(Dataset):
    def __init__(self, checkpoint_dirs: List[str], pattern: str = "step*.pkl"):
        # Ensure checkpoint_dirs is a list
        if isinstance(checkpoint_dirs, str):
            checkpoint_dirs = [checkpoint_dirs]
            
        # Collect files from all directories
        files = []
        for checkpoint_dir in checkpoint_dirs:
            dir_files = sorted(glob.glob(os.path.join(checkpoint_dir, pattern)))[2:]  # Skip first checkpoints
            files.extend(dir_files)
            
        if len(files) < 2:
            raise ValueError(f"Need at least 2 checkpoints, found {len(files)} across {len(checkpoint_dirs)} directories")
        
        # Load first checkpoint from first directory to get matrix shapes
        with open(files[0], 'rb') as f:
            first_checkpoint = pickle.load(f)
        
        self.samples = [MatrixSample(name, shape, file) 
                       for file in files 
                       for name, _, shape in first_checkpoint]

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.samples[idx]
        with open(sample.file_path, 'rb') as f:
            params = pickle.load(f)
            for name, matrix, _ in params:
                if name == sample.name:
                    return torch.from_numpy(matrix)
        raise ValueError(f"Matrix {sample.name} not found in {sample.file_path}")

    def __len__(self) -> int:
        return len(self.samples)

def create_dataloader(checkpoint_dirs: List[str], batch_size: int = 8, rank: int = 0, world_size: int = 1) -> DataLoader:
    dataset = MatrixDataset(checkpoint_dirs)
    shape_indices = {}
    
    for idx, sample in enumerate(dataset.samples):
        shape_indices.setdefault(sample.shape, []).append(idx)
    
    # Adjust batch size based on world size to avoid OOM
    local_batch_size = max(1, batch_size // world_size)  # Ensure at least 1 sample per batch
    
    # Create a DistributedSampler-like class for balanced shape-aware distribution
    class ShapeDistributedSampler(torch.utils.data.Sampler):
        def __init__(self):
            self.epoch = 0  # Track current epoch
            
        def set_epoch(self, epoch):
            """Set the epoch for this sampler to ensure different shuffling per epoch"""
            self.epoch = epoch
            
        def __iter__(self):
            # Process each shape group separately to maintain shape consistency in batches
            all_indices = []
            
            # Use epoch-dependent seed for consistent but different shuffling each epoch
            base_seed = 42
            seed = base_seed + self.epoch
            
            for shape, indices in shape_indices.items():
                # Use deterministic shuffle with epoch-dependent seed
                g = torch.Generator()
                g.manual_seed(seed)
                
                # Shuffle indices for this shape
                indices_tensor = torch.tensor(indices, dtype=torch.int64)
                perm = torch.randperm(len(indices), generator=g)
                indices_shuffled = indices_tensor[perm].tolist()
                
                # Pad to make divisible by (world_size * local_batch_size)
                # This ensures each process gets equal number of samples
                padding_size = (world_size * local_batch_size) - (len(indices_shuffled) % (world_size * local_batch_size))
                if padding_size < world_size * local_batch_size:
                    # Cycle through indices to pad
                    padding = indices_shuffled[:padding_size]
                    indices_padded = indices_shuffled + padding
                else:
                    indices_padded = indices_shuffled
                
                # Reshape to [num_batches, world_size, local_batch_size]
                num_samples = len(indices_padded)
                num_batches = num_samples // (world_size * local_batch_size)
                
                # Reshape to distribute evenly across processes
                indices_reshaped = torch.tensor(indices_padded).view(num_batches, world_size, local_batch_size)
                
                # Extract batches for this rank
                for i in range(num_batches):
                    # Get this rank's batch for the i-th global batch
                    batch = indices_reshaped[i, rank].tolist()
                    all_indices.append(batch)
            
            # Shuffle the batches with epoch-dependent seed
            g = torch.Generator()
            g.manual_seed(seed + 1)  # Different seed than above but still epoch-dependent
            batch_perm = torch.randperm(len(all_indices), generator=g)
            all_indices = [all_indices[i] for i in batch_perm.tolist()]
            
            # Return batches assigned to this rank
            return iter(all_indices)
            
        def __len__(self):
            # Count total batches for this rank
            total_samples = sum(len(indices) for indices in shape_indices.values())
            # Add padding to make divisible
            padded_total = total_samples
            for indices in shape_indices.values():
                padding_size = (world_size * local_batch_size) - (len(indices) % (world_size * local_batch_size))
                if padding_size < world_size * local_batch_size:
                    padded_total += padding_size
            
            return padded_total // (world_size * local_batch_size)
    
    # Create sampler instance
    sampler = ShapeDistributedSampler()
    
    # Pin memory for faster GPU transfer
    return {
        'dataloader': DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=lambda x: torch.stack(x),
            num_workers=2,  # Reduced from 4 to save memory
            pin_memory=True,
            persistent_workers=True  # Keep workers alive between epochs
        ),
        'sampler': sampler  # Return sampler to allow setting epoch
    }

def naive_loss(model_output):
    I = torch.eye(model_output.size(-1), device=model_output.device)
    # Use a more stable version of the loss calculation
    diff = model_output.transpose(-2, -1) @ model_output - (1 - 1e-8) * I
    
    # Normalize by the number of elements in the result matrix (which is n×n)
    # X can be m×n (rectangular), but X^T X is always n×n
    n = model_output.size(-1)
    
    # Return normalized Frobenius norm
    return torch.norm(diff, p='fro') / n

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
        init_coefficients=init_coefficients,
        scaling_factor=config.get('scaling_factor', 32.0)  # Add scaling factor
    ).to(device)
    
    # Broadcast model parameters from rank 0 to all processes
    for param in model.parameters():
        dist.broadcast(param.data, 0)
    
    # Create parameter groups with different learning rates
    # Regular parameters get the base learning rate
    # Last two quintic polynomials get a higher learning rate (5x)
    param_groups = [
        {'params': [p for i, p in enumerate(model.poly_layers) if i < len(model.poly_layers) - 2], 
         'lr': config['learning_rate']},
        {'params': [p for i, p in enumerate(model.poly_layers) if i >= len(model.poly_layers) - 2], 
         'lr': config['learning_rate'] * 5.0}  # 5x higher learning rate for last two layers
    ]
    
    # Log the learning rates if master process
    if master_process:
        log(f"Base learning rate: {config['learning_rate']}", console=True)
        log(f"Last two polynomial learning rate: {config['learning_rate'] * 5.0}", console=True)
        log(f"Using scaling factor: {config.get('scaling_factor', 32.0)}", console=True)
    
    optimizer = torch.optim.Adam(
        param_groups,
        betas=config['adam_betas']
    )
    
    # Create dataloader with adjusted batch size
    dataloader_dict = create_dataloader(
        config['checkpoint_dirs'], 
        config['batch_size'],
        rank=rank,
        world_size=world_size
    )
    dataloader = dataloader_dict['dataloader']
    sampler = dataloader_dict['sampler']
    
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
    
    for epoch in range(config['num_epochs']):
        # Set epoch for the sampler to ensure different shuffling each epoch
        sampler.set_epoch(epoch)
        
        epoch_loss = 0
        num_batches = 0
        current_loss = 0
        epoch_start_time = time.perf_counter()

        # Add a small delay between epochs to ensure all processes are synchronized
        if epoch > 0:
            torch.cuda.synchronize()
            dist.barrier()

        for i in range(1000):
        # for i, matrices in enumerate(dataloader):
            # Get current elapsed time
            current_elapsed_time = time.perf_counter() - start_time
            univariate_loss_value = univariate_loss(model, device)
            derivative_reward_value = model.derivative_at_zero()
            loss = config['univariate_loss_weight'] * univariate_loss_value - config['derivative_reward_weight'] * derivative_reward_value
            
            if master_process and i % 10 == 0:
                # log(f"Epoch {epoch}, batch {i}: naive loss: {naive_loss_value.item():.4e}, "
                    # f"derivative: {derivative_reward_value.item():.4e}, univariate loss: {univariate_loss_value.item():.4e}, Elapsed: {current_elapsed_time:.2f}s", console=True)
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
            
            optimizer.zero_grad(set_to_none=True)
            
            current_loss += loss.item()
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log to wandb periodically
            if (i + 1) % config['accumulation_steps'] == 0 and master_process:
                wandb.log({
                    "loss": current_loss / config['accumulation_steps'],
                    # "naive_loss": naive_loss_value.item(),
                    "derivative": derivative_reward_value.item(),
                    "univariate_loss": univariate_loss_value.item()
                })
                current_loss = 0
            
            # Free memory
            # del matrices
            torch.cuda.empty_cache() if config.get('aggressive_memory_cleanup', False) else None

        # Log epoch statistics
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start_time
        total_train_time += epoch_time
        
        if master_process:
            log(f"Epoch {epoch}/{config['num_epochs']} completed in {epoch_time:.2f}s, avg_loss: {epoch_loss/max(1, num_batches):.6f}", console=True)
            wandb.log({
                "epoch": epoch,
                "avg_epoch_loss": epoch_loss / max(1, num_batches),
                "epoch_time": epoch_time,
                "polynomial": plot_fn(model.evaluate_scalar),
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
        
        # Finish wandb run
        if run is not None:
            run.finish()
    
    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    default_config = {
        "device": "cuda",
        "checkpoint_dirs": ["matrices/a519aa14-cfde-4f80-aa58-87c9d9761a2a"],
        "learning_rate": 1e-5,
        "adam_betas": (0.8, 0.8),
        "batch_size": 64,  # Global batch size
        "batch_size_per_gpu": 8,  # Per-GPU batch size (will override global batch size)
        "accumulation_steps": 3,
        "max_grad_norm": 1e-3,
        "num_epochs": 10000,
        "derivative_reward_weight": 1e-6,
        "univariate_loss_weight": 1000,
        "save_checkpoints": True,
        "use_compile": True,  # Use torch.compile
        "aggressive_memory_cleanup": False,  # Enable for extreme OOM cases
        "scaling_factor": 32.0,  # Add scaling factor for bfloat16 training
        "init_coefficients": [
            [1.5, -1.0, 0.0],
            [1.5, -1.0, 0.0],
            [1.5, -1.0, 0.0],
            [1.5, -1.0, 0.0],
            [1.5, -1.0, 0.0],
            [1.5, -1.0, 0.0],
            [1.5, -1.0, 0.0],
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