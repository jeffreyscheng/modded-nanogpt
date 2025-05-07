import os
import sys
import uuid
import time
import copy
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch._logging._internal import trace_structured # noqa: E402
import torch._inductor.codecache # noqa: E402
import torch._inductor.graph # noqa: E402

with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
torch._inductor.config.coordinate_descent_tuning = True # we allow this flag for medium track
torch._dynamo.config.compiled_autograd = True

# Import all the functions and classes from gpt_static.py
from gpt_static import (
    zeropower_via_newtonschulz5, update, norm, init_linear, next_multiple_of_n,
    _load_data_shard, distributed_data_generator, get_lr, get_window_size_blocks_helper,
    get_window_size_blocks, nvidia_smi, print0, opt_params, 
    Muon, Rotary, CausalSelfAttention, MLP, Block, GPT, Hyperparameters
)

def load_checkpoint(checkpoint_path, device="cuda", rank=0):
    """
    Load a model checkpoint from the given path.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        rank: Process rank (for distributed training)
        
    Returns:
        model: The loaded model
        step: The training step of the checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    args = Hyperparameters()
    
    # Create a new model instance
    model = GPT(vocab_size=args.vocab_size, num_layers=16, num_heads=8, model_dim=1024,
                max_seq_len=max(args.train_seq_len, args.val_seq_len)).to(device)
    
    # Convert model parameters to bfloat16
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()
    
    # Load the checkpoint
    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    step = checkpoint.get('step', 0)
    
    # Fix state dict keys - remove '_orig_mod.' prefix for compiled models
    model_state_dict = checkpoint['model']
    if any(key.startswith('_orig_mod.') for key in model_state_dict.keys()):
        if rank == 0:
            print("Detected compiled model checkpoint, fixing keys...")
        fixed_state_dict = {}
        for key, value in model_state_dict.items():
            if key.startswith('_orig_mod.'):
                fixed_key = key[len('_orig_mod.'):]
                fixed_state_dict[fixed_key] = value
            else:
                fixed_state_dict[key] = value
        model_state_dict = fixed_state_dict
    
    # Ensure all tensor values are in bfloat16 format
    for key, value in model_state_dict.items():
        if isinstance(value, torch.Tensor) and value.dtype != torch.bfloat16 and value.is_floating_point():
            model_state_dict[key] = value.to(torch.bfloat16)
    
    # Load state dict with strict=False to allow for parameter differences
    model.load_state_dict(model_state_dict, strict=False)
    
    # Ensure all model parameters are in bfloat16
    for param_name, param in model.named_parameters():
        if param.is_floating_point() and param.dtype != torch.bfloat16:
            if rank == 0:
                print(f"Converting parameter {param_name} from {param.dtype} to bfloat16")
            param.data = param.data.to(torch.bfloat16)
    
    # Synchronize model parameters across ranks if in distributed mode
    if dist.is_initialized() and dist.get_world_size() > 1:
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)
    
    return model, step

def extract_model_parameters(model):
    """
    Extract model parameters as a dictionary.
    
    Args:
        model: The model to extract parameters from
        
    Returns:
        dict: Dictionary mapping parameter names to tensors
    """
    return {name: param.clone().detach() for name, param in model.named_parameters()}

def interpolate_models(models, weights):
    """
    Interpolate between multiple models using the given weights.
    
    Args:
        models: List of models to interpolate between
        weights: List of weights for each model (should sum to 1)
        
    Returns:
        model: A new model with interpolated parameters
    """
    if len(models) != len(weights):
        raise ValueError("Number of models must match number of weights")
    
    # Verify weights sum to approximately 1
    weight_sum = sum(weights)
    if not (0.99 <= weight_sum <= 1.01):
        print(f"Warning: Weights sum to {weight_sum}, not 1.0. Normalizing weights.")
        weights = [w / weight_sum for w in weights]
    
    # Create a new model with the same architecture as the first model
    interpolated_model = copy.deepcopy(models[0])
    
    # Extract parameters from each model
    model_params = [extract_model_parameters(model) for model in models]
    
    # Interpolate parameters
    with torch.no_grad():
        for name, param in interpolated_model.named_parameters():
            # Initialize parameter with zeros
            param.zero_()
            
            # Add weighted contribution from each model
            for i, weight in enumerate(weights):
                if weight > 0:
                    param.add_(model_params[i][name] * weight)
    
    return interpolated_model

def compute_validation_loss(model, rank=0, world_size=1, device="cuda", seed=42):
    """
    Compute the validation loss for a given model.
    
    Args:
        model: The model to evaluate
        rank: Process rank (for distributed training)
        world_size: Total number of processes (for distributed training)
        device: Device to run validation on
        seed: Random seed for validation data
        
    Returns:
        float: The validation loss
    """
    args = Hyperparameters()
    
    # Set model to evaluation mode
    model.eval()
    
    # Configure validation parameters
    val_batch_size = world_size * args.val_seq_len
    
    # Calculate validation steps using the same formula as in training
    val_steps = args.val_tokens // val_batch_size
    
    # Create validation data generator
    val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size, seed=seed)
    
    # Compute validation loss
    val_loss = 0
    with torch.no_grad():
        for _ in range(val_steps):
            inputs, targets = next(val_loader)
            # Ensure inputs are int32 as expected by the model
            if inputs.dtype != torch.int32:
                inputs = inputs.to(torch.int32)
            
            # Forward pass with the correct window size blocks
            val_loss += model(inputs, targets, get_window_size_blocks(0))
    
    val_loss /= val_steps
    
    # Average loss across all processes if in distributed mode
    if dist.is_initialized() and world_size > 1:
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    
    return val_loss.item()

def train_model(seed=0, run_id=0, iterations=None, checkpoint_path=None, output_dir=None):
    # Set PyTorch random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -----------------------------------------------------------------------------
    # int main

    args = Hyperparameters()
    
    # Use custom iterations if provided
    if iterations is None:
        iterations = args.num_iterations

    # torchrun sets these env variables
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 8 # this code is designed for 8xH100
    assert torch.cuda.is_available()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    # Skip init_process_group as it's now handled by the caller
    # dist.init_process_group(backend="nccl", device_id=device)
    master_process = (rank == 0) # this process will do logging, checkpointing etc.

    # Set up output directory for final model
    if output_dir is None:
        output_dir = "logs"
        
    # Determine if output_dir is a directory or a complete filepath
    is_filepath = output_dir.endswith('.pt')
    
    # If it's a directory, create it
    if not is_filepath and master_process and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # begin logging
    if master_process:
        # Include seed in the run_id_full to differentiate runs
        run_id_full = f"{run_id:03d}_seed{seed}_{uuid.uuid4()}"
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{run_id_full}.txt"
        print(logfile)
    def print0_local(s, console=False):
        if master_process:
            with open(logfile, "a") as f:
                if console:
                    print(s)
                print(s, file=f)
    def _patched_trace_structured(name, metadata_fn, **kwargs):
        if name == "inductor_output_code":
            print0_local(f"inductor_output_code: {metadata_fn().get('filename', 'Unknown')}")
        trace_structured(name, metadata_fn, **kwargs)
    torch._inductor.codecache.trace_structured = _patched_trace_structured
    torch._inductor.graph.trace_structured = _patched_trace_structured

    # begin by printing this file (the Python code)
    print0_local(code)
    print0_local("="*100)
    # log information about the hardware/software environment this is running on
    print0_local(f"Running Python {sys.version}")
    print0_local(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
    print0_local(f"Using random seed: {seed}")
    if checkpoint_path:
        print0_local(f"Loading checkpoint from: {checkpoint_path}")
    print0_local(f"Number of iterations: {iterations}")
    print0_local(f"Output directory: {output_dir}")
    print0_local(nvidia_smi())
    print0_local("="*100)

    ########################################
    #    Construct model and optimizer     #
    ########################################

    model: torch.nn.Module = GPT(vocab_size=args.vocab_size, num_layers=16, num_heads=8, model_dim=1024,
                        max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda()
    for m in model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.bfloat16()
    # Sync model parameters across ranks after initial creation
    for param in model.parameters():
        dist.broadcast(param.detach(), 0)

    # collect the parameters to optimize
    hidden_matrix_params = sorted((p for p in model.blocks.parameters() if p.ndim >= 2), key=lambda x: x.size(), reverse=True)
    embed_params = [*model.embed.parameters(), *model.value_embeds.parameters()]
    scalar_params = [model.scalars]
    head_params: list[torch.nn.Parameter] = [model.lm_head_w]
    # sanity check
    params_collections = [hidden_matrix_params, embed_params, scalar_params, head_params]
    optimized_parameters_set = {p for params in params_collections for p in params}
    assert optimized_parameters_set == {*model.parameters()}
    assert len(optimized_parameters_set) == sum(len(lst) for lst in params_collections)

    # init the optimizer(s)
    adam_param_groups = [dict(params=head_params, lr=1/320), dict(params=embed_params, lr=0.3), dict(params=scalar_params, lr=0.015)]
    # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
    # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
    optimizer1 = torch.optim.AdamW(adam_param_groups, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, fused=True)
    optimizer2 = Muon(hidden_matrix_params, lr=0.025, momentum=0.95, rank=rank, world_size=world_size)
    optimizers: list[torch.optim.Optimizer] = [optimizer1, optimizer2]
    opt2params = {opt: opt_params(opt) for opt in optimizers}
    for opt in optimizers:
        for group in opt.param_groups:
            group["initial_lr"] = group["lr"]

    # DON'T compile the model yet - this was the issue!
    # model: torch.nn.Module = torch.compile(model, dynamic=False)

    # Load from checkpoint if provided
    start_step = 0
    loaded_from_checkpoint = False
    if checkpoint_path and os.path.exists(checkpoint_path):
        loaded_from_checkpoint = True
        if master_process:
            print0_local(f"Loading checkpoint from {checkpoint_path}", console=True)
        
        # Load checkpoint on rank 0 to avoid file system contention
        if rank == 0:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'step' in checkpoint:
                start_step = checkpoint['step']
            else:
                print0_local("Warning: Checkpoint doesn't contain 'step' key, starting from step 0", console=True)
        else:
            checkpoint = None
            
        # Broadcast start_step from rank 0 to all processes
        start_step_tensor = torch.tensor([start_step], dtype=torch.long, device='cuda')
        dist.broadcast(start_step_tensor, 0)
        start_step = int(start_step_tensor.item())
        
        # If checkpoint was loaded, apply to model and optimizers
        if rank == 0 and checkpoint is not None:
            # Fix state dict keys - remove '_orig_mod.' prefix for compiled models
            model_state_dict = checkpoint['model']
            if any(key.startswith('_orig_mod.') for key in model_state_dict.keys()):
                print0_local("Detected compiled model checkpoint, fixing keys...", console=True)
                fixed_state_dict = {}
                for key, value in model_state_dict.items():
                    if key.startswith('_orig_mod.'):
                        fixed_key = key[len('_orig_mod.'):]
                        fixed_state_dict[fixed_key] = value
                    else:
                        fixed_state_dict[key] = value
                model_state_dict = fixed_state_dict
            
            # Load state dict with strict=False to allow for parameter differences
            model.load_state_dict(model_state_dict, strict=False)
            
            # Load optimizer states
            for i, opt in enumerate(optimizers):
                if i < len(checkpoint.get('optimizers', [])):
                    opt.load_state_dict(checkpoint['optimizers'][i])
                    
        # Synchronize model parameters across all ranks
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)
            
        # Add checkpoint loading verification
        if master_process:
            print0_local("Verifying checkpoint loading...", console=True)
            
        # Verify no parameter is a zero matrix
        zero_params = []
        for name, param in model.named_parameters():
            # Check if parameter is all zeros
            is_all_zeros = (param.abs().sum().item() == 0)
            if is_all_zeros:
                zero_params.append(name)
                
        if zero_params:
            raise ValueError(f"Found {len(zero_params)} zero matrices after loading the checkpoint: {zero_params[:5]}")
        
        # Verify parameter norms are reasonable
        if master_process:
            sample_norms = {name: param.norm().item() for name, param in list(model.named_parameters())[:5]}
            print0_local(f"Sample parameter norms: {sample_norms}", console=True)
            
        if master_process:
            print0_local(f"Resuming from step {start_step}", console=True)
    
    # Calculate end step based on iterations parameter
    # If iterations is small (like 1-10), treat it as relative to start_step
    # Otherwise treat it as an absolute iteration count
    if iterations < 100 and start_step > 0:
        end_step = start_step + iterations
        print0_local(f"Will train for {iterations} more iterations (from {start_step} to {end_step})", console=True)
    else:
        end_step = iterations
        print0_local(f"Will train until iteration {end_step} (currently at {start_step})", console=True)
        
    # Handle case where we're just loading a checkpoint and not training further
    if start_step >= end_step:
        if master_process:
            print0_local(f"No additional training needed (start_step={start_step}, end_step={end_step})", console=True)
            # Save the model to output_dir
            is_filepath = output_dir.endswith('.pt')
            final_checkpoint_path = output_dir if is_filepath else f"{output_dir}/final_model_seed{seed}_step{start_step:06d}.pt"
            os.makedirs(os.path.dirname(final_checkpoint_path), exist_ok=True)
            log = dict(step=start_step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            torch.save(log, final_checkpoint_path)
            print0_local(f"Saved loaded model to {final_checkpoint_path}", console=True)
        # Don't destroy process group here, let the caller handle it
        return 0.0  # Return dummy validation loss

    # NOW compile the model AFTER loading the checkpoint
    model: torch.nn.Module = torch.compile(model, dynamic=False)

    ########################################
    #            Warmup kernels            #
    ########################################

    # Skip warmup if loading from checkpoint
    if not loaded_from_checkpoint:
        # Warmup the training kernels, then re-initialize the state so we aren't cheating
        warmup_steps = 10
        initial_state = copy.deepcopy(dict(model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers]))
        # Use the seed for warmup data as well
        torch.manual_seed(seed + 5000)  # Different seed offset for warmup
        for _ in range(warmup_steps):
            inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
            model(inputs.to(torch.int32), targets, get_window_size_blocks(0)).backward()
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            for opt in optimizers:
                opt.step()
            model.zero_grad(set_to_none=True)
        model.load_state_dict(initial_state["model"])
        for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
            opt.load_state_dict(opt_state)
        del initial_state
        # Ensure model parameters are in sync after warmup
        for param in model.parameters():
            dist.broadcast(param.detach(), 0)
        # Reset the seed after warmup
        torch.manual_seed(seed)
    else:
        print0_local("Skipping warmup steps since loading from checkpoint", console=True)
        # Ensure optimizer states are properly initialized for the loaded model
        model.zero_grad(set_to_none=True)
        for opt in optimizers:
            if isinstance(opt, Muon):
                # Reinitialize Muon optimizer state for each parameter
                for group in opt.param_groups:
                    for p in group['params']:
                        # Make sure parameters are bfloat16
                        if p.dtype != torch.bfloat16:
                            p.data = p.data.to(torch.bfloat16)
                        
                        # Clear existing state if any
                        if p in opt.state:
                            del opt.state[p]
                        
                        # Correctly set up optimizer state with empty tensors
                        opt.state[p] = {}  # Start with empty state

    ########################################
    #        Training and validation       #
    ########################################

    torch.cuda.reset_peak_memory_stats()
    # Pass the seed to the distributed_data_generator
    train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size, seed=seed)
    training_time_ms = 0
    # start the clock
    # dist.barrier() # Removed, now handled by the caller
    t0 = time.perf_counter()
    # begin training
    train_steps = end_step
    for step in range(start_step, train_steps + 1):
        last_step = (step == train_steps)

        # --------------- VALIDATION SECTION -----------------
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            # stop the clock
            # dist.barrier() # Removed, now handled by the caller
            training_time_ms += 1000 * (time.perf_counter() - t0)
            model.eval()
            val_batch_size = world_size * args.val_seq_len
            assert args.val_tokens % val_batch_size == 0
            val_steps = args.val_tokens // val_batch_size
            # Also pass the seed to validation data loader
            val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size, seed=seed + 10000)  # Different seed for validation
            val_loss = 0
            with torch.no_grad():
                for _ in range(val_steps):
                    inputs, targets = next(val_loader)
                    val_loss += model(inputs, targets, get_window_size_blocks(step))
            val_loss /= val_steps
            del val_loader
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            print0_local(f"step:{step}/{train_steps} val_loss:{val_loss:.6f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step - start_step, 1):.2f}ms", console=True)
            model.train()
            # start the clock again
            # dist.barrier() # Removed, now handled by the caller
            t0 = time.perf_counter()

        if last_step:
            if master_process:
                final_checkpoint_path = f"{output_dir}/final_model_seed{seed}_step{step:06d}.pt"
                os.makedirs(os.path.dirname(final_checkpoint_path), exist_ok=True)    
                log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
                torch.save(log, final_checkpoint_path)
                print0_local(f"Saved final model to {final_checkpoint_path}", console=True)
            # the last step only has the validation loop, so break to avoid training
            break

        # --------------- TRAINING SECTION -----------------
        inputs, targets = next(train_loader)
        model(inputs, targets, get_window_size_blocks(step)).backward()
        opt2futures = {
            opt: [dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True).get_future() for p in params]
            for opt, params in opt2params.items()
        }
        # set optimization hyperparameters
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * get_lr(step)
        for group in optimizer2.param_groups:
            frac = min(step / 300, 1) # momentum warmup for muon
            group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
        # step the optimizers
        for opt in optimizers:
            torch.futures.collect_all(opt2futures[opt]).wait()
            opt.step()
        # null the gradients
        model.zero_grad(set_to_none=True)
        # logging
        approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
        print0_local(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step - start_step + 1):.2f}ms", console=True)

    print0_local(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
    # Don't destroy process group here, let the caller handle it
    return val_loss # Return final validation loss

if __name__ == "__main__":
    # Initialize the process group once for all runs
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    
    try:
        starting_minibatch_step = 2000
        forward_iterations = 1
        checkpoint_path = f"/home/paperspace/dev/modded-nanogpt/logs/gptm_record/state_step{starting_minibatch_step:06d}.pt"
        
        # Create the parent directory for output files if it doesn't exist
        output_parent_dir = "/home/paperspace/dev/modded-nanogpt/souping_logs"
        os.makedirs(output_parent_dir, exist_ok=True)
        
        for seed in range(8):
            print(f"Starting seed {seed}")
            output_dir = f"{output_parent_dir}/step{starting_minibatch_step:06d}_seed{seed}_forward{forward_iterations}.pt"
            train_model(seed=seed, run_id=0, iterations=forward_iterations, 
                        checkpoint_path=checkpoint_path, output_dir=output_dir)
            # synchronize
            dist.barrier()
            
        # Only destroy the process group once, at the very end
        dist.destroy_process_group()
    except Exception as e:
        # Clean up even if there's an error
        if dist.is_initialized():
            dist.destroy_process_group()
        raise e