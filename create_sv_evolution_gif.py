import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import re
import glob
from tqdm import tqdm
import imageio.v2 as imageio
from pathlib import Path
from collections import defaultdict

# Path to checkpoints
BASE_PATH = "gradient_logs/029747bb-dc0c-4e55-a7b3-43b9c1955e23"
OUTPUT_DIR = "sv_evolution_frames"
GIF_PATH = "sv_evolution.gif"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_checkpoint_data(checkpoint_path):
    """Process checkpoint data and return processed DataFrame"""
    csv_path = os.path.join(checkpoint_path, "minibatch_noise.csv")
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found at {csv_path}")
        return None
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Apply the function to create a new column with processed weight names
        df['weight_type'] = df['weight_name']
        
        # Filter out non-block weights
        df_filtered = df[df['weight_type'].notna()]
        
        # Check if there are any weight types
        if len(df_filtered['weight_type'].unique()) == 0:
            # Try a different pattern if the standard one didn't work
            alt_pattern = r'blocks\.\d+\.([a-zA-Z0-9_\.]+)'
            df['weight_type'] = df['weight_name'].apply(lambda x: re.search(alt_pattern, str(x)).group(1) if isinstance(x, str) and re.search(alt_pattern, str(x)) else None)
            df_filtered = df[df['weight_type'].notna()]
            if len(df_filtered['weight_type'].unique()) == 0:
                return None
        
        # Compute normalized values
        # df_filtered['sv_mean'] = df_filtered['singular_value_mean'] / df_filtered['frobenius_norm']
        # df_filtered['sv_std'] = df_filtered['singular_value_variance'].apply(np.sqrt) / df_filtered['frobenius_norm']
        df_filtered['sv_mean'] = df_filtered['singular_value_mean']
        df_filtered['sv_std'] = df_filtered['singular_value_variance'].apply(np.sqrt)
        
        return df_filtered
    
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")
        return None

def compute_axis_bounds(valid_checkpoint_dirs):
    """Compute consistent axis bounds for each weight type across all checkpoints"""
    print("Computing consistent axis bounds across all checkpoints...")
    
    # Dictionary to store min/max values for each weight type
    bounds = defaultdict(lambda: {'x_min': float('inf'), 'x_max': float('-inf'), 
                                 'y_min': float('inf'), 'y_max': float('-inf')})
    
    # Dictionary to store all unique weight types
    all_weight_types = set()
    
    # Process each checkpoint to find global min/max values
    for checkpoint_dir in tqdm(valid_checkpoint_dirs):
        df_filtered = process_checkpoint_data(checkpoint_dir)
        
        if df_filtered is not None:
            # Update the set of all unique weight types
            weight_types = df_filtered['weight_type'].unique()
            all_weight_types.update(weight_types)
            
            # For each weight type, update the min/max values
            for weight_type in weight_types:
                weight_df = df_filtered[df_filtered['weight_type'] == weight_type]
                
                # Get min/max values, filtering out inf, -inf, NaN
                sv_mean = weight_df['sv_mean'].replace([np.inf, -np.inf], np.nan).dropna()
                sv_std = weight_df['sv_std'].replace([np.inf, -np.inf], np.nan).dropna()
                
                if not sv_mean.empty and not sv_std.empty:
                    # Filter out non-positive values for log scale
                    sv_mean_positive = sv_mean[sv_mean > 0]
                    sv_std_positive = sv_std[sv_std > 0]
                    
                    if not sv_mean_positive.empty:
                        bounds[weight_type]['x_min'] = min(bounds[weight_type]['x_min'], sv_mean_positive.min())
                        bounds[weight_type]['x_max'] = max(bounds[weight_type]['x_max'], sv_mean.max())
                    
                    if not sv_std_positive.empty:
                        bounds[weight_type]['y_min'] = min(bounds[weight_type]['y_min'], sv_std_positive.min())
                        bounds[weight_type]['y_max'] = max(bounds[weight_type]['y_max'], sv_std.max())
    
    # Add some padding to the bounds (important for log scale)
    for weight_type in bounds:
        # Ensure minimum values are positive
        bounds[weight_type]['x_min'] = max(1e-10, bounds[weight_type]['x_min'] * 0.9)
        bounds[weight_type]['x_max'] *= 1.1
        bounds[weight_type]['y_min'] = max(1e-10, bounds[weight_type]['y_min'] * 0.9)
        bounds[weight_type]['y_max'] *= 1.1
    
    return bounds, list(all_weight_types)

def create_frame(checkpoint_path, frame_number, axis_bounds, all_weight_types):
    """Create a visualization frame for a single checkpoint with consistent axis bounds"""
    print(f"Processing checkpoint: {checkpoint_path}")
    
    try:
        # Process the checkpoint data
        df_filtered = process_checkpoint_data(checkpoint_path)
        if df_filtered is None:
            return None
        
        # Get weight types present in this checkpoint
        checkpoint_weight_types = df_filtered['weight_type'].unique()
        
        # Calculate grid dimensions based on all possible weight types
        num_types = len(all_weight_types)
        num_cols = int(np.ceil(np.sqrt(num_types)))
        num_rows = int(np.ceil(num_types / num_cols))
        
        # Create a figure with subplots, leave room for colorbar on the right
        fig = plt.figure(figsize=(5*num_cols + 1, 4*num_rows))
        gs = fig.add_gridspec(num_rows, num_cols + 1, width_ratios=[1]*num_cols + [0.1])
        
        # Create a list to hold all axes
        axes = []
        for i in range(num_rows):
            row_axes = []
            for j in range(num_cols):
                ax = fig.add_subplot(gs[i, j])
                row_axes.append(ax)
            axes.append(row_axes)
        
        # Convert to NumPy array for easier indexing
        axes = np.array(axes)
        axes_flat = axes.flatten()
        
        # Create a colormap for consistent coloring across subplots based on layer number
        layer_min = df_filtered['layer_number'].min()
        layer_max = df_filtered['layer_number'].max()
        norm = Normalize(vmin=layer_min, vmax=layer_max)
        cmap = plt.cm.get_cmap('viridis')
        
        # Create a placeholder for the scatter plot object to use for the colorbar
        scatter_for_cbar = None
        
        # Create subplots for each weight type (always in the same position)
        for i, weight_type in enumerate(all_weight_types):
            if i < len(axes_flat):  # Ensure we don't exceed the number of subplots
                ax = axes_flat[i]
                
                # Set scales
                ax.set_xscale('log')
                ax.set_yscale('log')
                
                # Apply consistent axis limits for this weight type
                if weight_type in axis_bounds:
                    bounds = axis_bounds[weight_type]
                    ax.set_xlim(bounds['x_min'], bounds['x_max'])
                    ax.set_ylim(bounds['y_min'], bounds['y_max'])
                
                # Add title and labels
                ax.set_title(f'{weight_type}', fontsize=10)
                
                # Add labels only to the leftmost and bottom subplots
                if i % num_cols == 0:  # Leftmost subplot in each row
                    ax.set_ylabel('Std Dev of SV')
                if i >= num_cols * (num_rows - 1):  # Bottom row
                    ax.set_xlabel('Mean of SV')
                
                # Check if this weight type is present in the current checkpoint
                if weight_type in checkpoint_weight_types:
                    # Filter data for this weight type
                    weight_df = df_filtered[df_filtered['weight_type'] == weight_type]
                    
                    # Plot data
                    scatter = ax.scatter(weight_df['sv_mean'], weight_df['sv_std'], 
                                        alpha=0.1, c=weight_df['layer_number'],
                                        cmap=cmap, norm=norm)
                    
                    # Save the scatter object for the colorbar reference
                    if scatter_for_cbar is None:
                        scatter_for_cbar = scatter
                    
                    # Add a count of the number of layers represented
                    layer_count = len(weight_df['layer_number'].unique())
                    ax.annotate(f'Layers: {layer_count}', xy=(0.05, 0.95), xycoords='axes fraction',
                               fontsize=8, ha='left', va='top')
                else:
                    # Add a note that this weight type is not present in this checkpoint
                    ax.annotate('No data in this checkpoint', xy=(0.5, 0.5), xycoords='axes fraction',
                               fontsize=10, ha='center', va='center', alpha=0.5)
        
        # Add colorbar at the right side of the figure
        if scatter_for_cbar is not None:
            cbar_ax = fig.add_subplot(gs[:, -1])  # Use the last column of gridspec for colorbar
            cbar = fig.colorbar(scatter_for_cbar, cax=cbar_ax)
            cbar.set_label('Layer Number')
            cbar.solids.set_alpha(1.0)  # Make the colorbar fully opaque
        
        # Extract checkpoint number from path
        checkpoint_num = os.path.basename(checkpoint_path)
        
        # Add an overall title
        fig.suptitle(f'Variance of Gradient Singular Values over Minibatch Seeds, Minibatch {checkpoint_num}', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        fig.subplots_adjust(top=0.93)  # Adjust to make room for suptitle
        
        # Save the figure to a file
        output_path = os.path.join(OUTPUT_DIR, f"frame_{frame_number:04d}.png")
        plt.savefig(output_path, dpi=120)
        plt.close(fig)
        
        return output_path
    except Exception as e:
        print(f"Error processing {checkpoint_path}: {str(e)}")
        return None

def main():
    # First, check for valid checkpoint directories (those with CSV files)
    valid_checkpoint_dirs = []
    
    print(f"Scanning for valid checkpoints in {BASE_PATH}...")
    all_checkpoint_dirs = glob.glob(os.path.join(BASE_PATH, "*"))
    for checkpoint_dir in all_checkpoint_dirs:
        csv_path = os.path.join(checkpoint_dir, "minibatch_noise.csv")
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            valid_checkpoint_dirs.append(checkpoint_dir)
    
    # Sort checkpoints by number
    valid_checkpoint_dirs = sorted(
        valid_checkpoint_dirs, 
        key=lambda x: int(os.path.basename(x)) if os.path.basename(x).isdigit() else float('inf')
    )
    
    print(f"Found {len(valid_checkpoint_dirs)} valid checkpoint directories with CSV files")
    
    if len(valid_checkpoint_dirs) == 0:
        print("No valid checkpoints found with CSV files. Cannot create GIF.")
        return
    
    # Compute consistent axis bounds for all weight types across all checkpoints
    axis_bounds, all_weight_types = compute_axis_bounds(valid_checkpoint_dirs)
    print(f"Found {len(all_weight_types)} unique weight types across all checkpoints")
    
    # Create a frame for each checkpoint using consistent axis bounds
    frames = []
    for i, checkpoint_dir in enumerate(tqdm(valid_checkpoint_dirs)):
        frame_path = create_frame(checkpoint_dir, i, axis_bounds, all_weight_types)
        if frame_path:
            frames.append(frame_path)
    
    # Create GIF
    if frames:
        print(f"Creating GIF with {len(frames)} frames")
        with imageio.get_writer(GIF_PATH, mode='I', fps=15, loop=0) as writer:
            for frame_path in frames:
                image = imageio.imread(frame_path)
                writer.append_data(image)
        print(f"GIF saved to {GIF_PATH}")
    else:
        print("No frames were created, cannot create GIF")

if __name__ == "__main__":
    main() 