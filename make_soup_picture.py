import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_gptm_short import compute_validation_loss, Hyperparameters, GPT, get_window_size_blocks, distributed_data_generator
import torch.distributed as dist
import gc

def load_checkpoint_local(checkpoint_path, device="cpu"):
    """
    Load a model checkpoint without using distributed operations.
    Returns the state dict and step only.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    step = checkpoint.get('step', 0)
    
    # Fix state dict keys - remove '_orig_mod.' prefix for compiled models
    model_state_dict = checkpoint['model']
    
    # Create a new state dict with standardized keys
    fixed_state_dict = {}
    for key, value in model_state_dict.items():
        # Remove any '_orig_mod.' prefix
        fixed_key = key
        if key.startswith('_orig_mod.'):
            fixed_key = key[len('_orig_mod.'):]
            
        # Convert parameters to bfloat16 if needed
        if isinstance(value, torch.Tensor) and value.dtype != torch.bfloat16 and value.is_floating_point():
            value = value.to(torch.bfloat16)
            
        fixed_state_dict[fixed_key] = value
    
    return fixed_state_dict, step

def load_state_dict_safely(model, state_dict, strict=False):
    """
    Safely load a state dictionary into a model, handling prefixes correctly.
    """
    # Get model keys
    model_keys = set(k for k, _ in model.named_parameters())
    
    # Create a fixed state dict with matching keys
    fixed_state_dict = {}
    for key, value in state_dict.items():
        # Try to find the correct key for this parameter
        if key in model_keys:
            # Direct match
            fixed_state_dict[key] = value
        elif key.startswith('_orig_mod.') and key[len('_orig_mod.'):] in model_keys:
            # Remove prefix
            fixed_key = key[len('_orig_mod.'):]
            fixed_state_dict[fixed_key] = value
        elif f"_orig_mod.{key}" in model_keys:
            # Add prefix
            fixed_key = f"_orig_mod.{key}"
            fixed_state_dict[fixed_key] = value
        else:
            # Best effort - keep original key
            fixed_state_dict[key] = value
    
    # Load the state dict
    model.load_state_dict(fixed_state_dict, strict=strict)
    
    # Return success
    return True

def make_soup_picture(
    early_path, late_path1, late_path2, 
    output_path="model_soup_viz.png", 
    grid_size=10, world_size=1, rank=0, device="cuda"
):
    """Create a visualization of loss landscape by interpolating between three checkpoints"""
    if rank == 0:
        print(f"Loading checkpoints on CPU...")
    
    # Load model state dicts on CPU without using distributed operations directly
    if rank == 0:
        # Only rank 0 loads the checkpoints
        state_dict_early, step_early = load_checkpoint_local(early_path, device="cpu")
        state_dict_late1, step_late1 = load_checkpoint_local(late_path1, device="cpu")
        state_dict_late2, step_late2 = load_checkpoint_local(late_path2, device="cpu")
    else:
        # Other ranks just initialize placeholder variables
        step_early, step_late1, step_late2 = 0, 0, 0
    
    # Create common model architecture on each rank
    args = Hyperparameters()
    
    # Every rank creates the same model structure
    eval_model = GPT(vocab_size=args.vocab_size, num_layers=16, num_heads=8, model_dim=1024,
                     max_seq_len=max(args.train_seq_len, args.val_seq_len)).to(device)
    
    # Convert all parameters to bfloat16
    for m in eval_model.modules():
        if isinstance(m, torch.nn.Embedding):
            m.weight.data = m.weight.data.to(torch.bfloat16)
    
    for param in eval_model.parameters():
        if param.is_floating_point() and param.dtype != torch.bfloat16:
            param.data = param.data.to(torch.bfloat16)
    
    # Compile the model on each rank
    eval_model = torch.compile(eval_model, dynamic=False)
    
    # Broadcast step values to all ranks
    if world_size > 1:
        step_tensor = torch.tensor([step_early, step_late1, step_late2], dtype=torch.long, device=device)
        dist.broadcast(step_tensor, 0)
        if rank != 0:
            step_early, step_late1, step_late2 = step_tensor.tolist()
    
    # Create grid for interpolation
    alpha = np.linspace(0, 1, grid_size)
    beta = np.linspace(0, 1, grid_size)
    loss_grid = np.zeros((grid_size, grid_size))
    
    # Evaluate each grid point
    total_points = grid_size * grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            a, b = alpha[i], beta[j]
            w_early = max(0, 1 - a - b)
            w_late1, w_late2 = a, b
            
            # Skip points outside the triangle
            if w_early < 0:
                loss_grid[j, i] = float('nan')
                continue
                
            if rank == 0:
                print(f"Point {i*grid_size+j+1}/{total_points}: ({a:.2f}, {b:.2f})")
                
                # Get all possible keys, standardizing them
                all_keys_early = set(state_dict_early.keys())
                all_keys_late1 = set(state_dict_late1.keys())
                all_keys_late2 = set(state_dict_late2.keys())
                
                # Find common keys (without prefixes)
                common_keys = set()
                key_mapping = {}  # Maps standardized key to actual keys in each dict
                
                # Map all keys
                for key in all_keys_early:
                    std_key = key[len('_orig_mod.'):] if key.startswith('_orig_mod.') else key
                    key_mapping[std_key] = {
                        'early': key, 
                        'late1': next((k for k in all_keys_late1 if k == std_key or k.endswith(f".{std_key}") or k == f"_orig_mod.{std_key}"), None),
                        'late2': next((k for k in all_keys_late2 if k == std_key or k.endswith(f".{std_key}") or k == f"_orig_mod.{std_key}"), None)
                    }
                    if key_mapping[std_key]['late1'] and key_mapping[std_key]['late2']:
                        common_keys.add(std_key)
                
                # Now create the interpolated state dict
                state_dict = {}
                for std_key in common_keys:
                    mapping = key_mapping[std_key]
                    
                    # Get the tensors using the actual keys
                    tensor_early = state_dict_early[mapping['early']]
                    tensor_late1 = state_dict_late1[mapping['late1']]
                    tensor_late2 = state_dict_late2[mapping['late2']]
                    
                    # Verify tensor shapes match
                    if tensor_early.shape != tensor_late1.shape or tensor_early.shape != tensor_late2.shape:
                        continue
                        
                    # Interpolate
                    interpolated = (
                        w_early * tensor_early + 
                        w_late1 * tensor_late1 + 
                        w_late2 * tensor_late2
                    )
                    
                    # Verify the interpolated tensor is not all zeros
                    if torch.all(interpolated == 0):
                        # Use the non-zero tensor
                        if tensor_early.norm().item() > 0:
                            interpolated = tensor_early
                        elif tensor_late1.norm().item() > 0:
                            interpolated = tensor_late1
                        elif tensor_late2.norm().item() > 0:
                            interpolated = tensor_late2
                    
                    # Use the original key format from the model parameters
                    model_key = std_key
                    for name, _ in eval_model.named_parameters():
                        if name == std_key or name.endswith(f".{std_key}"):
                            model_key = name
                            break
                            
                    state_dict[model_key] = interpolated
                
                # Load interpolated state dict
                load_state_dict_safely(eval_model, state_dict, strict=False)
            
            # Sync parameters across ranks (from rank 0 to all)
            for param in eval_model.parameters():
                dist.broadcast(param.data, 0)
                
            # Force sync before proceeding
            torch.cuda.synchronize()
            if world_size > 1:
                dist.barrier()
            
            # Evaluate model
            eval_model.eval()
            val_loss = compute_validation_loss(
                eval_model, rank=rank, 
                world_size=world_size, device=device,
                seed=42 + i*grid_size + j
            )
            
            loss_grid[j, i] = val_loss
            if rank == 0:
                print(f"  Loss: {val_loss:.4f}")
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Force sync before next iteration
            if world_size > 1:
                dist.barrier()
    
    # Create visualization (rank 0 only)
    if rank == 0:
        plt.figure(figsize=(10, 8))
        mesh_alpha, mesh_beta = np.meshgrid(alpha, beta)
        
        # Plot contour
        vertices = np.array([[0, 0], [1, 0], [0, 1]])
        labels = ["Early", f"Late 1 (step {step_late1})", f"Late 2 (step {step_late2})"]
        
        contour = plt.contourf(mesh_alpha, mesh_beta, loss_grid, 20, cmap='viridis')
        plt.colorbar(contour, label='Validation Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Mark vertices
        plt.scatter(vertices[:, 0], vertices[:, 1], c='red', s=100, zorder=5)
        for i, (x, y) in enumerate(vertices):
            plt.annotate(labels[i], (x, y), xytext=(10 if x > 0.5 else -10, 10 if y > 0.5 else -10), 
                        textcoords='offset points', ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        
        # Draw triangle boundary
        plt.gca().add_patch(plt.Polygon(vertices, fill=False, edgecolor='red', linestyle='--'))
        
        # Add labels
        plt.xlabel('Weight of Late Model 1')
        plt.ylabel('Weight of Late Model 2')
        plt.title('Validation Loss Landscape for Model Interpolation')
        
        # Mark minimum loss point
        valid_indices = ~np.isnan(loss_grid)
        if np.any(valid_indices):
            min_loss = np.nanmin(loss_grid)
            min_idx = np.where(loss_grid == min_loss)
            min_a, min_b = alpha[min_idx[1][0]], beta[min_idx[0][0]]
            
            plt.scatter(min_a, min_b, c='yellow', s=200, marker='*', edgecolor='black', zorder=6)
            plt.annotate(f"Min Loss: {min_loss:.4f}\nat ({min_a:.2f}, {min_b:.2f})",
                        (min_a, min_b), xytext=(15, 15), textcoords='offset points',
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Saved visualization to {output_path}")
    
    return loss_grid

if __name__ == "__main__":
    # Configure paths
    early_path = "/home/paperspace/dev/modded-nanogpt/logs/gptm_record/state_step002000.pt"
    late_path1 = "/home/paperspace/dev/modded-nanogpt/souping_logs/step002000_seed0_forward1.pt/final_model_seed0_step002001.pt"
    late_path2 = "/home/paperspace/dev/modded-nanogpt/souping_logs/step002000_seed1_forward1.pt/final_model_seed1_step002001.pt"
    output_path = "model_soup_viz.png"
    grid_size = 10  # Default grid size
    
    # Set up distributed environment
    rank, world_size = 0, 1
    device = "cuda"
    
    # Use all available GPUs with torch.distributed
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        
        # Print GPU information
        if rank == 0:
            gpu_count = torch.cuda.device_count()
            print(f"Available GPUs: {gpu_count}")
            for i in range(gpu_count):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            if rank == 0:
                print(f"Initialized distributed process group with world_size={world_size}")
    else:
        # When running without torchrun, suggest the proper command
        print("Not running in distributed mode.")
        print("To use all 8 GPUs, run with: source nanogpt-venv/bin/activate && python -m torch.distributed.run --nproc_per_node=8 make_soup_picture.py")
    
    # Run the soup picture generation
    if rank == 0:
        print(f"Starting visualization with grid size {grid_size}x{grid_size}, world_size={world_size}")
    
    make_soup_picture(
        early_path, late_path1, late_path2,
        output_path=output_path,
        grid_size=grid_size,
        world_size=world_size,
        rank=rank,
        device=device
    )
    
    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()
    
    if rank == 0:
        print("Done!")