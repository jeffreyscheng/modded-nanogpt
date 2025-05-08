import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from train_gptm_short import Hyperparameters, GPT, get_window_size_blocks, distributed_data_generator, run_validation
import torch.distributed as dist
import gc
import pandas as pd
from itertools import product
from matplotlib.colors import PowerNorm

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
    """Safely load a state dictionary into a model, handling prefixes correctly."""
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

def calculate_loss_landscape(
    record_name, step, forward_iterations, early_path, late_path1, late_path2, 
    grid_size=10, world_size=1, rank=0, device="cuda", margin=0.2
):
    """Calculate loss landscape by interpolating between three checkpoints
    
    Returns a DataFrame with columns: record_name, step, forward_iterations, a, b, val_loss
    """
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
    
    # Create grid for interpolation with margin
    alpha = np.linspace(0 - margin, 1 + margin, grid_size)
    beta = np.linspace(0 - margin, 1 + margin, grid_size)
    loss_data = []
    
    # Original triangle vertices
    orig_vertices = np.array([[0, 0], [1, 0], [0, 1]])
    # Expanded triangle vertices with margin
    expanded_vertices = np.array([
        [0 - margin, 0 - margin], 
        [1 + margin, 0 - margin], 
        [0 - margin, 1 + margin]
    ])
    
    # Evaluate each grid point
    total_valid_points = 0
    for i in range(grid_size):
        for j in range(grid_size):
            # Count all points in the expanded triangle
            a, b = alpha[i], beta[j]
            # We'll evaluate points where a + b <= 1 + 2*margin (expanded triangle)
            if a + b <= 1 + 2*margin:
                total_valid_points += 1
    
    if rank == 0:
        print(f"Calculating {total_valid_points} points in the expanded triangle")
    
    # Process all valid points
    point_count = 0
    for i in range(grid_size):
        a = alpha[i]
        for j in range(grid_size):
            b = beta[j]
            # Skip points outside the expanded triangle
            if a + b > 1 + 2*margin:
                continue
                
            # Calculate interpolation weights, ensuring they sum to 1
            w_early = 1 - a - b
            w_late1, w_late2 = a, b
            
            point_count += 1
            
            if rank == 0:
                print(f"Point {point_count}/{total_valid_points}: ({a:.2f}, {b:.2f})")
                
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
            
            # Define dummy variables needed for run_validation function
            seed = 42 + i*grid_size + j  # Use a unique seed for each point
            val_loss = run_validation(eval_model, step, args, rank, world_size, seed)
            
            # Store the point data
            loss_data.append({
                'record_name': record_name,
                'step': step,
                'forward_iterations': forward_iterations,
                'a': a,
                'b': b,
                'val_loss': val_loss.item()
            })
            
            if rank == 0:
                print(f"  Loss: {val_loss.item():.4f}")
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Force sync before next iteration
            if world_size > 1:
                dist.barrier()
    
    # Create DataFrame
    df = pd.DataFrame(loss_data)
    return df

def plot_soupy_landscape(dataframes, output_dir="souping_imgs", margin=0.2):
    """Create a compound plot from multiple loss landscape dataframes
    
    Args:
        dataframes: List of dataframes with loss landscape data
        output_dir: Directory to save the plots
        margin: Margin value for the expanded triangle
    """
    if not dataframes:
        return
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract unique values for steps and forward_iterations
    all_df = pd.concat(dataframes)
    unique_steps = sorted(all_df['step'].unique())
    unique_forward_iters = sorted(all_df['forward_iterations'].unique(), reverse=True)
    unique_record_names = sorted(all_df['record_name'].unique())
    
    # For each record_name, create a grid of plots
    for record_name in unique_record_names:
        record_df = all_df[all_df['record_name'] == record_name]
        
        # Skip if no data for this record
        if record_df.empty:
            continue
            
        # Calculate per-configuration min/max values for better plotting
        config_min_max = {}
        for step in unique_steps:
            for forward_iter in unique_forward_iters:
                subset = record_df[(record_df['step'] == step) & 
                                  (record_df['forward_iterations'] == forward_iter)]
                if not subset.empty:
                    config_min_max[(step, forward_iter)] = (subset['val_loss'].min(), subset['val_loss'].max())
        
        # Get global min and max, but with some padding to ensure visible contours
        all_mins = [v[0] for v in config_min_max.values()]
        all_maxs = [v[1] for v in config_min_max.values()]
        global_min = min(all_mins) if all_mins else None
        global_max = max(all_maxs) if all_maxs else None
        
        # Ensure there's sufficient range for visible contours
        if global_min is not None and global_max is not None:
            # If range is too small, expand it
            if global_max - global_min < 0.01:
                mid_point = (global_min + global_max) / 2
                global_min = mid_point - 0.01
                global_max = mid_point + 0.01
        
        # Create figure with a grid of subplots
        # rows = number of forward_iterations, columns = number of steps
        fig, axes = plt.subplots(
            nrows=len(unique_forward_iters),
            ncols=len(unique_steps),
            figsize=(4*len(unique_steps), 3.5*len(unique_forward_iters)),
            squeeze=False,
            facecolor='white'
        )
        
        # Set up plot style
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 10,
            'axes.facecolor': 'white',
            'axes.edgecolor': '#dddddd',
            'axes.labelcolor': '#555555',
            'xtick.color': '#555555',
            'ytick.color': '#555555',
            'grid.color': '#f2f2f2'
        })
        
        # Use a standard continuous colormap more sensitive to low values
        cmap = plt.cm.hsv  # Reversed viridis (dark blue for low values, yellow for high)
        
        # Create a normalization that emphasizes lower values
        if global_min is not None and global_max is not None:
            norm = PowerNorm(gamma=0.5, vmin=global_min, vmax=global_max)
        else:
            norm = None
        
        # Original triangle vertices
        orig_vertices = np.array([[0, 0], [1, 0], [0, 1]])
        # Expanded triangle vertices with margin
        expanded_vertices = np.array([
            [0 - margin, 0 - margin], 
            [1 + margin, 0 - margin], 
            [0 - margin, 1 + margin]
        ])
        
        # For tracking shared colorbar
        shared_contour = None
        
        # Plot each subplot
        for row_idx, forward_iter in enumerate(unique_forward_iters):
            for col_idx, step in enumerate(unique_steps):
                ax = axes[row_idx, col_idx]
                
                # Get data for this specific combination
                subplot_df = record_df[(record_df['step'] == step) & 
                                      (record_df['forward_iterations'] == forward_iter)]
                
                if subplot_df.empty:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                
                # Grid size for this subplot
                grid_size = int(np.sqrt(len(subplot_df['a'].unique())))
                alpha = np.sort(subplot_df['a'].unique())
                beta = np.sort(subplot_df['b'].unique())
                
                # Create 2D grid for contour plot
                mesh_alpha, mesh_beta = np.meshgrid(alpha, beta)
                
                # Create the loss grid
                loss_grid = np.full((len(beta), len(alpha)), np.nan)
                for idx, row in subplot_df.iterrows():
                    a_idx = np.where(alpha == row['a'])[0][0]
                    b_idx = np.where(beta == row['b'])[0][0]
                    loss_grid[b_idx, a_idx] = row['val_loss']
                
                # Get local min/max for this subplot
                local_min = np.nanmin(loss_grid)
                local_max = np.nanmax(loss_grid)
                
                # Create contour levels that will show variation within this subplot
                # Using local min/max with padding to ensure visible contours
                level_padding = max(0.0001, (local_max - local_min) * 0.1)  # At least 0.0001 or 10% of range
                local_levels = np.linspace(
                    local_min - level_padding,
                    local_max + level_padding,
                    10  # More levels for finer gradation
                )
                
                # Use global min/max for consistent coloring, but local levels for contour lines
                contour = ax.contourf(mesh_alpha, mesh_beta, loss_grid, 
                                     levels=local_levels,
                                     cmap=cmap, 
                                     norm=norm,
                                     extend='both')
                
                # Store first contour for shared colorbar
                if shared_contour is None:
                    shared_contour = contour
                
                # Add contour lines for better visibility
                ax.contour(mesh_alpha, mesh_beta, loss_grid, 
                          levels=local_levels,
                          colors='black', 
                          linewidths=0.5, 
                          alpha=0.3)
                
                # Add subtle grid
                ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
                
                # Mark original vertices with markers
                ax.scatter(orig_vertices[:, 0], orig_vertices[:, 1], c='#FF5E78', s=80, 
                          zorder=5, marker='o', edgecolor='white', linewidth=1.5)
                
                # Draw expanded triangle boundary
                ax.add_patch(plt.Polygon(expanded_vertices, fill=False, 
                                        edgecolor='#aaaaaa', linestyle=':', linewidth=0.8, alpha=0.6))
                
                # Add simplified labels to original vertices
                labels = [
                    f"$\\theta_{{{step}}}$",
                    f"$\\theta_{{{step+forward_iter}}}^{{1}}$", 
                    f"$\\theta_{{{step+forward_iter}}}^{{2}}$"
                ]
                
                # Add annotations with minimal style
                for i, (x, y) in enumerate(orig_vertices):
                    offset_x = 12 if x > 0.5 else -12
                    offset_y = 12 if y > 0.5 else -12
                    ax.annotate(labels[i], (x, y), 
                               xytext=(offset_x, offset_y), 
                               textcoords='offset points', 
                               ha='center', va='center',
                               bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='#FF5E78', alpha=0.9),
                               fontsize=9)
                
                # Draw original triangle boundary
                ax.add_patch(plt.Polygon(orig_vertices, fill=False, edgecolor='#FF5E78', 
                                        linestyle='-', linewidth=1.5, alpha=0.7))
                
                # Set title with forward iterations and step
                ax.set_title(f"Step {step}, Forward {forward_iter}", fontsize=10, color='#555555')
                
                # Mark minimum loss point with a star
                valid_indices = ~np.isnan(loss_grid)
                if np.any(valid_indices):
                    min_loss_local = np.nanmin(loss_grid)
                    min_idx = np.where(loss_grid == min_loss_local)
                    if len(min_idx[0]) > 0:  # Check if any minimum found
                        min_a, min_b = alpha[min_idx[1][0]], beta[min_idx[0][0]]
                        
                        ax.scatter(min_a, min_b, c='#FFDE00', s=180, marker='*', 
                                  edgecolor='#555555', zorder=6, linewidth=1)
                        ax.annotate(f"{min_loss_local:.4f}",
                                   (min_a, min_b), 
                                   xytext=(15, 10), 
                                   textcoords='offset points',
                                   bbox=dict(boxstyle="round,pad=0.2", fc='#FFDE00', ec='#555555', alpha=0.9),
                                   arrowprops=dict(arrowstyle="-", color="#888888", shrinkA=5, shrinkB=5, 
                                                  connectionstyle="arc3,rad=.1"),
                                   fontsize=9)
                
                # Remove axis labels and ticks for minimalism
                ax.set_xticks([])
                ax.set_yticks([])
        
        # Add row and column labels
        for i, forward_iter in enumerate(unique_forward_iters):
            fig.text(0.02, 0.5 + (i - len(unique_forward_iters)/2 + 0.5) / len(unique_forward_iters), 
                    f"Forward {forward_iter}", 
                    va='center', ha='center', rotation=90, fontsize=12)
            
        for j, step in enumerate(unique_steps):
            fig.text(0.5 + (j - len(unique_steps)/2 + 0.5) / len(unique_steps), 0.02, 
                    f"Step {step}", 
                    va='center', ha='center', fontsize=12)
        
        # Add colorbar using the shared contour
        if shared_contour is not None:
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            cbar = fig.colorbar(shared_contour, cax=cbar_ax)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label('Loss', size=10, labelpad=8)
            
            # Add min/max labels to colorbar
            if global_min is not None and global_max is not None:
                fig.text(0.95, 0.15, f"Min: {global_min:.4f}", fontsize=8, ha='right')
                fig.text(0.95, 0.85, f"Max: {global_max:.4f}", fontsize=8, ha='right')
        
        # Overall title
        fig.suptitle(f'Parameter Interpolation Loss Landscape - {record_name}', 
                    fontsize=14, color='#555555', y=0.98)
        
        plt.tight_layout(rect=[0.03, 0.03, 0.9, 0.95])
        
        # Save figure
        output_path = os.path.join(output_dir, f"{record_name}_compound.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Saved compound visualization to {output_path}")

if __name__ == "__main__":
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

    # Set up distributed environment
    rank, world_size = 0, 8
    device = "cuda"

    # Configure paths
    record_names = ["gptm_adam", "gptm_record"]
    starting_minibatch_steps = [125, 4000, 2000]
    forward_iteration_values = [200, 100, 50, 20, 10, 1]
    margin = 0.2
    grid_size = 20
    
    # Storage for all dataframes
    all_dataframes = []
    
    # Calculate loss landscapes for all combinations
    for record_name, step, forward_iterations in product(record_names, starting_minibatch_steps, forward_iteration_values):
        early_path = f"logs/{record_name}/state_step{step:06d}.pt"
        late_path1 = f"souping_logs/{record_name}_step{step:06d}/seed0_forward{forward_iterations}.pt"
        late_path2 = f"souping_logs/{record_name}_step{step:06d}/seed1_forward{forward_iterations}.pt"
        
        if rank == 0:
            print(f"Starting data collection for {record_name}, step {step}, forward {forward_iterations}")
        
        # Calculate loss landscape
        df = calculate_loss_landscape(
            record_name=record_name,
            step=step,
            forward_iterations=forward_iterations,
            early_path=early_path,
            late_path1=late_path1,
            late_path2=late_path2,
            grid_size=grid_size,
            world_size=world_size,
            rank=rank,
            device=device,
            margin=margin
        )
        
        # Save to list if on rank 0
        if rank == 0:
            all_dataframes.append(df)
            
            # Also save individual dataframe
            os.makedirs("souping_data", exist_ok=True)
            df.to_csv(f"souping_data/{record_name}_step{step}_forward{forward_iterations}.csv", index=False)
    
    all_dataframes = [pd.read_csv(f"souping_data/{f}") for f in os.listdir("souping_data") if f.endswith('.csv')]
    # Create compound plots for each record type
    if rank == 0 and all_dataframes:
        plot_soupy_landscape(all_dataframes, margin=margin)
    
    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()
    
    if rank == 0:
        print("Done!")