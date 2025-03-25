import os
import re
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import imageio
from typing import Dict, List, Tuple, Any
import torch
from tqdm import tqdm
from collections import defaultdict

# Normalize and compute singular values
def compute_singular_values(matrix: torch.Tensor) -> np.ndarray:
    """Compute singular values efficiently using torch SVD on GPU if available."""
    if matrix.device.type == 'cpu' and torch.cuda.is_available():
        matrix = matrix.cuda()
    
    # Convert to float32 for SVD operations
    if matrix.dtype != torch.float32:
        matrix = matrix.to(torch.float32)
        
    # Normalize by Frobenius norm
    norm = torch.linalg.norm(matrix)
    if norm > 0:
        matrix = matrix / norm
    
    # Use torch.linalg.svdvals which is faster than full SVD
    with torch.no_grad():
        s = torch.linalg.svdvals(matrix)
    return s.cpu().numpy()

def categorize_parameter(param_name: str) -> Tuple[str, int]:
    """Categorize parameter by type and layer number."""
    layer_match = re.search(r'blocks\.(\d+)', param_name)
    layer_num = int(layer_match.group(1)) if layer_match else -1
    
    if 'attn.c_proj.weight' in param_name:
        param_type = 'Attention Output'
    elif 'attn.qkv_w' in param_name:
        param_type = 'QKV'
    elif 'mlp.c_fc.weight' in param_name:
        param_type = 'MLP Input'
    elif 'mlp.c_proj.weight' in param_name:
        param_type = 'MLP Output'
    elif 'embed' in param_name:
        param_type = 'Embedding'
    elif 'lm_head' in param_name:
        param_type = 'LM Head'
    else:
        param_type = 'Other'
        
    return param_type, layer_num

def create_histogram_frame(singular_values: Dict, title: str, stacked: bool = False, 
                          log_scale: bool = True, axis_bounds: Dict = None) -> np.ndarray:
    """Create a histogram frame efficiently with fixed axis bounds."""
    # Use a smaller figure size for faster rendering
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    
    # Calculate bins once
    all_values = []
    if stacked:
        for category, values in singular_values.items():
            if isinstance(values, dict):
                for layer_values in values.values():
                    all_values.extend(layer_values)
            else:
                all_values.extend(values)
    else:
        all_values = singular_values['all']
        
    # Filter very small values and calculate bins
    all_values = [v for v in all_values if v > 1e-10]
    if not all_values:  # Prevent empty histogram errors
        all_values = [1e-9]
    
    # Set different number of bins depending on the visualization type
    if stacked and not isinstance(next(iter(singular_values.values()), None), dict):
        # For parameter type histograms, use more bins
        bins = 80  # Increased for smoother parameter type histograms
    else:
        bins = 50  # Slightly increased for layer histograms and overall histograms
    
    # Use pre-calculated bins if provided, otherwise calculate them
    if axis_bounds and 'bins' in axis_bounds:
        bins = axis_bounds['bins']
    elif log_scale and all_values:
        min_val = max(np.min(all_values), 1e-10)
        max_val = np.max(all_values)
        bins = np.logspace(np.log10(min_val), np.log10(max_val), bins)
    
    # Create the histogram based on data type
    if stacked:
        if isinstance(next(iter(singular_values.values())), dict):
            # Layer-specific stacking - only process layers with data
            for category, layer_data in singular_values.items():
                # Skip empty layers
                layers = [layer for layer, values in layer_data.items() if values]
                if not layers:
                    continue
                    
                # Sort layers
                layers.sort()
                
                # Always show individual layers instead of grouping
                layer_values = [layer_data[layer] for layer in layers]
                
                # Create a sequential colormap for layers - use a visually distinct colormap
                n_layers = len(layers)
                cmap = get_cmap('viridis')
                colors = [cmap(i / max(1, n_layers - 1)) for i in range(n_layers)]
                
                # Create labels for each layer
                labels = [f"Layer {layer}" for layer in layers]
                
                # Plot stacked histogram for this category
                ax.hist(layer_values, bins=bins, stacked=True, alpha=0.8, 
                       label=labels, color=colors)
                
                ax.set_title(f"{title} - {category}")
                
                # Add legend outside the plot for clarity
                ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        else:
            # Category stacking - only process categories with data
            categories = [cat for cat, values in singular_values.items() if values]
            if not categories:
                categories = ['No Data']
                singular_values = {'No Data': [0.0]}
                
            category_values = [singular_values[cat] for cat in categories]
            
            # Use a distinct colormap with high contrast
            cmap = get_cmap('tab10')
            colors = [cmap(i % 10) for i in range(len(categories))]
            
            # Plot stacked histogram
            ax.hist(category_values, bins=bins, stacked=True, alpha=0.8, 
                   label=categories, color=colors)
                   
            # Add legend outside the plot for clarity
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    else:
        # Simple histogram of all values
        ax.hist(singular_values['all'], bins=bins, alpha=0.8, color='steelblue')
    
    # Configure the plot
    if log_scale:
        ax.set_xscale('log')
    
    # Set fixed axis bounds if provided
    if axis_bounds:
        if 'xlim' in axis_bounds:
            ax.set_xlim(axis_bounds['xlim'])
        if 'ylim' in axis_bounds:
            ax.set_ylim(axis_bounds['ylim'])
    
    ax.set_title(title)
    ax.set_xlabel('Singular Value')
    ax.set_ylabel('Count')
    
    # Adjust figure size to accommodate the legend
    if stacked:
        plt.subplots_adjust(right=0.8)
    
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    # Fast conversion to numpy array with reduced memory usage
    fig.canvas.draw()
    
    # Get the RGBA buffer from the figure and convert to RGB
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape((h, w, 4))
    image = buf[:, :, :3]
    
    plt.close(fig)
    return image

def batch_compute_singular_values(matrices):
    """Process multiple matrices in a single batch for better performance."""
    if not matrices:
        return {}
        
    result = {}
    
    # Group matrices by shape for batch processing
    shape_groups = {}
    for name, matrix in matrices.items():
        if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
            continue
            
        shape = matrix.shape
        if shape not in shape_groups:
            shape_groups[shape] = []
        shape_groups[shape].append((name, matrix))
    
    # Process each shape group
    for shape, matrix_group in shape_groups.items():
        names = [item[0] for item in matrix_group]
        tensors = [item[1] for item in matrix_group]
        
        # Move to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Process each matrix individually to ensure proper normalization
        for name, tensor in zip(names, tensors):
            try:
                # Move to device and normalize
                tensor = tensor.to(device, dtype=torch.float32)
                
                # Normalize by Frobenius norm before computing SVD
                norm = torch.linalg.norm(tensor)
                if norm > 0:
                    tensor = tensor / norm
                
                singular_values = torch.linalg.svdvals(tensor).cpu().numpy()
                result[name] = singular_values
            except Exception as e:
                print(f"Error processing {name}: {e}")
    
    return result

def precompute_singular_values(pickle_files):
    """Precompute singular values for all matrices in all files."""
    all_singular_values = {}
    steps = []
    
    for pickle_file in tqdm(pickle_files, desc="Precomputing SVDs"):
        try:
            step = int(re.search(r'step(\d+)', pickle_file).group(1))
            steps.append(step)
            
            # Load matrices
            with open(pickle_file, 'rb') as f:
                matrices_dict = pickle.load(f)
            
            # Skip if no valid matrices
            if not isinstance(matrices_dict, dict) or not matrices_dict:
                continue
            
            # Compute singular values for all matrices
            singular_value_results = batch_compute_singular_values(matrices_dict)
            
            # Store results
            all_singular_values[step] = singular_value_results
            
        except Exception as e:
            print(f"Error precomputing SVD for {pickle_file}: {e}")
    
    return all_singular_values, sorted(steps)

def compute_stacked_histogram_counts(values_lists, bins):
    """
    Calculate the total height of a stacked histogram at each bin.
    
    Args:
        values_lists: List of arrays containing values for each stack
        bins: Histogram bins to use
    
    Returns:
        Array of the total heights at each bin
    """
    if not values_lists or not any(values_lists):
        return np.array([0])
    
    # Calculate histogram for each list
    hists = []
    for values in values_lists:
        if values:
            hist, _ = np.histogram(values, bins=bins)
            hists.append(hist)
    
    # Sum the histograms to get stacked height
    if hists:
        stacked_hist = np.sum(np.vstack(hists), axis=0)
        return stacked_hist
    
    return np.array([0])

def find_max_counts(all_singular_values, steps, param_types):
    """Find the maximum histogram count for each frame type across all steps."""
    # Track max counts for each visualization type
    max_overall_count = 0
    max_param_type_counts = 0  # Use a single max for param types
    max_layer_counts = {}
    
    # Process each step to find max counts
    for step in steps:
        if step not in all_singular_values:
            continue
            
        # Organize this step's data
        all_values = []
        param_type_values = {param_type: [] for param_type in param_types}
        type_layer_values = {param_type: {} for param_type in param_types}
        
        for name, singular_values in all_singular_values[step].items():
            all_values.extend(singular_values)
            
            param_type, layer_num = categorize_parameter(name)
            param_type_values[param_type].extend(singular_values)
            
            if layer_num not in type_layer_values[param_type]:
                type_layer_values[param_type][layer_num] = []
            type_layer_values[param_type][layer_num].extend(singular_values)
        
        # Calculate bins for this step
        min_val = max(np.min([v for v in all_values if v > 1e-10] or [1e-9]), 1e-10)
        max_val = np.max(all_values or [1.0])
        
        # Overall histogram count
        if all_values:
            overall_bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
            counts, _ = np.histogram([v for v in all_values if v > 1e-10], bins=overall_bins)
            max_overall_count = max(max_overall_count, np.max(counts))
        
        # Parameter type histogram count - use stacked counts
        if any(param_type_values.values()):
            param_bins = np.logspace(np.log10(min_val), np.log10(max_val), 80)
            param_values_list = [v for v in param_type_values.values() if v]
            stacked_counts = compute_stacked_histogram_counts(
                [[v for v in vals if v > 1e-10] for vals in param_values_list],
                param_bins
            )
            max_param_type_counts = max(max_param_type_counts, np.max(stacked_counts))
        
        # Layer-wise histogram counts - handle each parameter type separately
        for param_type, layer_data in type_layer_values.items():
            if not layer_data:
                continue
            
            # Get all values for this parameter type
            all_param_values = []
            for layer_values in layer_data.values():
                all_param_values.extend([v for v in layer_values if v > 1e-10])
            
            if all_param_values:
                layer_min = max(np.min(all_param_values), 1e-10)
                layer_max = np.max(all_param_values)
                layer_bins = np.logspace(np.log10(layer_min), np.log10(layer_max), 50)
                
                # Calculate stacked histogram for this parameter type
                layer_values_list = [
                    [v for v in values if v > 1e-10] 
                    for values in layer_data.values() if values
                ]
                
                stacked_counts = compute_stacked_histogram_counts(layer_values_list, layer_bins)
                current_max = np.max(stacked_counts)
                
                if param_type not in max_layer_counts:
                    max_layer_counts[param_type] = current_max
                else:
                    max_layer_counts[param_type] = max(max_layer_counts[param_type], current_max)
    
    return max_overall_count, max_param_type_counts, max_layer_counts

def determine_axis_bounds(all_singular_values, steps, param_types):
    """Determine consistent axis bounds for each visualization type."""
    # Get min/max singular values across all steps
    all_values_combined = []
    param_type_values_combined = {param_type: [] for param_type in param_types}
    type_layer_values_combined = {param_type: defaultdict(list) for param_type in param_types}
    
    # Collect all values to determine x-axis bounds
    for step in steps:
        if step not in all_singular_values:
            continue
            
        for name, values in all_singular_values[step].items():
            all_values_combined.extend(values)
            
            param_type, layer_num = categorize_parameter(name)
            param_type_values_combined[param_type].extend(values)
            type_layer_values_combined[param_type][layer_num].extend(values)
    
    # Filter very small values
    all_values_combined = [v for v in all_values_combined if v > 1e-10]
    
    # Get maximum histogram counts for each visualization type
    max_overall_count, max_param_type_counts, max_layer_counts = find_max_counts(
        all_singular_values, steps, param_types
    )
    
    # Compute global min/max for each visualization type
    axis_bounds = {}
    
    # Overall histogram bounds
    if all_values_combined:
        min_val = max(np.min(all_values_combined), 1e-10)
        max_val = np.max(all_values_combined)
        
        # Calculate histogram bins once
        overall_bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
        
        # Add 20% margin to y-axis
        y_max = max_overall_count * 1.2
        
        axis_bounds['overall'] = {
            'xlim': (min_val, max_val),
            'ylim': (0, y_max),
            'bins': overall_bins
        }
    
    # Parameter type histogram bounds
    if param_type_values_combined:
        min_vals = []
        max_vals = []
        for param_type, values in param_type_values_combined.items():
            filtered_values = [v for v in values if v > 1e-10]
            if filtered_values:
                min_vals.append(np.min(filtered_values))
                max_vals.append(np.max(filtered_values))
        
        if min_vals and max_vals:
            min_val = max(min(min_vals), 1e-10)
            max_val = max(max_vals)
            
            # Calculate histogram bins once for parameter type
            param_bins = np.logspace(np.log10(min_val), np.log10(max_val), 80)
            
            # Use the stacked height
            y_max = max_param_type_counts * 1.2  # Add 20% margin
            
            axis_bounds['param_type'] = {
                'xlim': (min_val, max_val),
                'ylim': (0, y_max),
                'bins': param_bins
            }
    
    # Layer-wise histogram bounds for each parameter type
    for param_type, layer_data in type_layer_values_combined.items():
        all_layer_values = []
        for layer_values in layer_data.values():
            all_layer_values.extend([v for v in layer_values if v > 1e-10])
        
        if all_layer_values:
            min_val = max(np.min(all_layer_values), 1e-10)
            max_val = np.max(all_layer_values)
            
            # Calculate histogram bins once for this parameter type's layers
            layer_bins = np.logspace(np.log10(min_val), np.log10(max_val), 50)
            
            # Use the max layer count for this parameter type with a generous margin
            if param_type in max_layer_counts:
                y_max = max_layer_counts[param_type] * 1.5  # Increased margin to 50% for more headroom
                
                axis_bounds[f'layer_{param_type}'] = {
                    'xlim': (min_val, max_val),
                    'ylim': (0, y_max),
                    'bins': layer_bins
                }
    
    return axis_bounds

def generate_visualizations(matrices_dir: str, output_dir: str, 
                           max_duration_sec: int = 10, fps: int = 30, 
                           batch_size: int = 5):
    """Generate all three visualization types with consistent axes."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'by_layer'), exist_ok=True)
    
    # Find and sort all pickle files
    pickle_files = sorted(glob.glob(os.path.join(matrices_dir, 'step*.pkl')), 
                         key=lambda x: int(re.search(r'step(\d+)', x).group(1)))
    
    if not pickle_files:
        print(f"No pickle files found in {matrices_dir}")
        return
    
    # Optimization: Skip files to maintain requested duration and fps
    total_frames = max_duration_sec * fps
    if len(pickle_files) > total_frames:
        # Select evenly spaced files
        indices = np.linspace(0, len(pickle_files)-1, total_frames, dtype=int)
        pickle_files = [pickle_files[i] for i in indices]
    
    print(f"Processing {len(pickle_files)} matrices out of total {len(glob.glob(os.path.join(matrices_dir, 'step*.pkl')))} files")
    
    # STEP 1: Identify all parameter types
    param_types = set()
    for file in pickle_files[:1]:
        try:
            with open(file, 'rb') as f:
                matrices = pickle.load(f)
                for name, matrix in matrices.items():
                    if isinstance(matrix, torch.Tensor) and matrix.ndim == 2:
                        param_type, _ = categorize_parameter(name)
                        param_types.add(param_type)
        except Exception as e:
            print(f"Error loading sample file {file}: {e}")
    
    if not param_types:
        print("No valid parameter types found. Check matrix files.")
        return
        
    print(f"Found parameter types: {', '.join(param_types)}")
    
    # STEP 2: Precompute all singular values
    print("Precomputing singular values for all matrices...")
    all_singular_values, steps = precompute_singular_values(pickle_files)
    
    if not all_singular_values:
        print("No valid singular values computed. Check matrix files.")
        return
    
    # STEP 3: Determine consistent axis bounds for all visualizations
    print("Determining axis bounds for consistent visualization...")
    axis_bounds = determine_axis_bounds(all_singular_values, steps, param_types)
    
    # STEP 4: Generate frames with consistent axis bounds
    print("Generating visualization frames with fixed axes...")
    
    # Initialize storage for all three visualization types
    all_sv_frames = []
    param_type_frames = []
    type_layer_frames = {param_type: [] for param_type in param_types}
    
    # Create frames for each step
    for step in tqdm(steps, desc="Generating frames"):
        if step not in all_singular_values or not all_singular_values[step]:
            continue
            
        # Organize results by type and layer
        all_values = []
        param_type_values = {param_type: [] for param_type in param_types}
        type_layer_values = {param_type: {} for param_type in param_types}
        
        # Organize the SVD results
        for name, singular_values in all_singular_values[step].items():
            all_values.extend(singular_values)
            
            param_type, layer_num = categorize_parameter(name)
            param_type_values[param_type].extend(singular_values)
            
            if layer_num not in type_layer_values[param_type]:
                type_layer_values[param_type][layer_num] = []
            type_layer_values[param_type][layer_num].extend(singular_values)
        
        # Create visualization frames with fixed axis bounds
        all_sv_frames.append(create_histogram_frame(
            {'all': all_values},
            f'Gradient Singular Values (Step {step})',
            axis_bounds=axis_bounds.get('overall')
        ))
        
        param_type_frames.append(create_histogram_frame(
            {k: v for k, v in param_type_values.items() if v},
            f'Gradient SV by Parameter Type (Step {step})',
            stacked=True,
            axis_bounds=axis_bounds.get('param_type')
        ))
        
        # Create type-layer frames
        for param_type, layer_data in type_layer_values.items():
            if not layer_data:
                continue
                
            frame = create_histogram_frame(
                {param_type: layer_data},
                f'{param_type} SV by Layer (Step {step})',
                stacked=True,
                axis_bounds=axis_bounds.get(f'layer_{param_type}')
            )
            type_layer_frames[param_type].append(frame)
    
    # STEP 5: Save GIFs
    if all_sv_frames:
        imageio.mimsave(os.path.join(output_dir, 'all_singular_values.gif'), 
                        all_sv_frames, fps=fps)
        print(f"Saved all_singular_values.gif with {len(all_sv_frames)} frames")
    else:
        print("Warning: No frames generated for all_singular_values.gif")
    
    if param_type_frames:
        imageio.mimsave(os.path.join(output_dir, 'parameter_types.gif'), 
                        param_type_frames, fps=fps)
        print(f"Saved parameter_types.gif with {len(param_type_frames)} frames")
    else:
        print("Warning: No frames generated for parameter_types.gif")
    
    for param_type, frames in type_layer_frames.items():
        if frames:
            output_path = os.path.join(output_dir, 'by_layer', 
                                      f'{param_type.replace(" ", "_")}.gif')
            imageio.mimsave(output_path, frames, fps=fps)
            print(f"Saved {param_type} layer-wise GIF with {len(frames)} frames")
    
    print("All visualizations complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate optimized visualizations of gradient singular values")
    parser.add_argument("matrices_dir", help="Directory with gradient matrix pickle files")
    parser.add_argument("--output-dir", default="visualizations", help="Directory for output GIFs")
    parser.add_argument("--max-duration", type=int, default=8, help="Maximum GIF duration in seconds")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second for GIFs")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of files to process in parallel")
    
    args = parser.parse_args()
    
    generate_visualizations(
        args.matrices_dir, 
        args.output_dir,
        max_duration_sec=args.max_duration,
        fps=args.fps,
        batch_size=args.batch_size
    )
