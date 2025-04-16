import torch
from train_gpt_classes import *
import os
import pandas as pd
import torch.distributed as dist
from pathlib import Path
import time
import copy
import numpy as np
import re
import gc
from typing import List, Dict, Tuple, Any

# Setup distributed environment
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "8"))
device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
torch.cuda.set_device(device)

# Global compiled model cache
_COMPILED_MODEL_CACHE = {}

if not dist.is_initialized():
    dist.init_process_group(backend="nccl")
master_process = (rank == 0)

def print0(message):
    """Print only from master process."""
    if master_process:
        print(message)

def get_available_checkpoints(run_id: str) -> List[int]:
    """
    Get a sorted list of all available checkpoint minibatches from the logs directory.
    
    Args:
        run_id: UUID of the training run
        
    Returns:
        Sorted list of available checkpoint step numbers, or empty list if none found
    """
    logs_dir = Path(f"logs/{run_id}")
    if not logs_dir.exists():
        print0(f"Logs directory 'logs/{run_id}' does not exist")
        return []
    
    # Get all checkpoint files
    checkpoint_files = list(logs_dir.glob("checkpoint_step*.pt"))
    
    # Extract minibatch numbers from filenames
    minibatches = []
    for file_path in checkpoint_files:
        match = re.search(r'checkpoint_step(\d+)\.pt$', file_path.name)
        if match:
            step_num = int(match.group(1))
            if step_num < 2970:
                continue
            minibatches.append(int(match.group(1)))
    
    print0(f"Found {len(minibatches)} checkpoints in logs/{run_id}")
    return sorted(minibatches)

def format_weight_name(name: str) -> str:
    """Format weight names to a cleaner format."""
    # Handle QKV split weights
    if name.endswith('_Q'): return 'attn.Q'
    if name.endswith('_K'): return 'attn.K'
    if name.endswith('_V'): return 'attn.V'
    
    # Handle other weights
    for pattern in ['mlp.c_fc', 'mlp.c_proj', 'attn.c_proj']:
        if pattern in name:
            return pattern
    
    return name  # Should not reach here for weights we care about

def is_essential_weight(name: str) -> bool:
    """Check if this is a weight we care about for SVD analysis."""
    if 'blocks' not in name:
        return False
        
    essential_patterns = ['mlp.c_fc', 'mlp.c_proj', 'attn.c_proj', 'attn.qkv']
    return any(pattern in name for pattern in essential_patterns)

def procrustes_align_gpu(X, Y):
    """Align matrix X to matrix Y using orthogonal Procrustes rotation on GPU."""
    dtype = X.dtype
    X_flat, Y_flat = X.reshape(X.shape[0], -1), Y.reshape(Y.shape[0], -1)
    
    # Convert to float32 for SVD
    X_flat_f32, Y_flat_f32 = X_flat.to(torch.float32), Y_flat.to(torch.float32)
    
    # Perform SVD in float32
    U, _, Vh = torch.linalg.svd((X_flat_f32.T @ Y_flat_f32), full_matrices=False)
    
    # Convert back to original dtype and perform the alignment
    return (X_flat @ (U.to(dtype) @ Vh.to(dtype))).reshape(X.shape)

def log_time(message, prev_time=None):
    """Helper to log timing with proper synchronization."""
    torch.cuda.synchronize()
    now = time.time()
    
    if master_process:
        elapsed = f" in {now - prev_time:.2f}s" if prev_time else ""
        print(f"Rank {rank}: {message}{elapsed}")
    
    return now

def get_param_matrices(model):
    """Extract parameter matrices from model, filtering to essential weights only."""
    param_matrices = {}
    for name, param in model.named_parameters():
        if param.ndim < 2 or not is_essential_weight(name):
            continue
            
        # Extract layer number
        layer_num = -1
        parts = name.split('.')
        for i, part in enumerate(parts):
            if part == 'blocks' and i+1 < len(parts) and parts[i+1].isdigit():
                layer_num = int(parts[i+1])
                break
        
        param_matrices[(layer_num, name)] = param.to(torch.bfloat16)
    return param_matrices

def split_qkv_tensors(gradients_dict: Dict) -> Dict:
    """Split QKV gradients into separate Q, K, V components."""
    split_dict = {}
    
    for key, grad in gradients_dict.items():
        if grad is None or grad.ndim < 2:
            continue
            
        # Check if this is a QKV gradient
        if 'qkv' in key[1].lower() and grad.shape[0] == 3:
            layer_num = key[0]
            
            # Split into Q, K, V components
            q_key = (layer_num, f"{key[1]}_Q")
            k_key = (layer_num, f"{key[1]}_K")
            v_key = (layer_num, f"{key[1]}_V")
            
            split_dict[q_key] = grad[0].clone()
            split_dict[k_key] = grad[1].clone()
            split_dict[v_key] = grad[2].clone()
        else:
            split_dict[key] = grad
            
    return split_dict

def process_svd_batch(keys, gradient_dict, svd_minibatch, log_dir=None):
    """Process a batch of SVD operations."""
    results = {}
    
    for i in range(0, len(keys), svd_minibatch):
        batch_keys = keys[i:i+svd_minibatch]
        
        if master_process and i % (svd_minibatch * 5) == 0:
            print(f"Processing SVD batch {i//svd_minibatch + 1}/{len(keys)//svd_minibatch + 1}")
        
        for key in batch_keys:
            if key not in gradient_dict or gradient_dict[key] is None or gradient_dict[key].ndim <= 1:
                continue
                
            try:
                grad = gradient_dict[key]
                
                # Convert to float32 for SVD
                grad_f32 = grad.to(torch.float32)
                
                # Perform SVD
                U, S, Vh = torch.linalg.svd(grad_f32, full_matrices=False)
                
                # Convert results back to bfloat16
                results[key] = (U.to(torch.bfloat16), S.to(torch.bfloat16), Vh.to(torch.bfloat16))
                
                # Save grad if log_dir provided
                if master_process and log_dir:
                    torch.save(grad, log_dir / f"layer_{key[0]}_weight_{key[1].replace('.', '_')}_avg.pt")
                    
            except Exception as e:
                print(f"SVD failed for key {key}: {e}")
        
        # Clear cache after each batch
        torch.cuda.empty_cache()
    
    return results

def cleanup_memory(*args):
    """Clean up memory by deleting variables and flushing cache."""
    for arg in args:
        del arg
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def get_or_create_compiled_model(model_cfg, force_recompile=False):
    """Get a cached compiled model or create and cache a new one.
    
    Args:
        model_cfg: Tuple of (vocab_size, num_layers, num_heads, model_dim, max_seq_len) for model config
        force_recompile: Whether to force recompilation even if already cached
        
    Returns:
        Compiled model
    """
    global _COMPILED_MODEL_CACHE
    
    # Check if we already have this model configuration cached
    if model_cfg in _COMPILED_MODEL_CACHE and not force_recompile:
        print0(f"Using cached compiled model for config {model_cfg}")
        return copy.deepcopy(_COMPILED_MODEL_CACHE[model_cfg])
    
    # Otherwise create and compile a new model
    print0(f"Creating and compiling new model for config {model_cfg}")
    
    vocab_size, num_layers, num_heads, model_dim, max_seq_len = model_cfg
    
    # Create model
    model = GPT(vocab_size=vocab_size, 
               num_layers=num_layers, 
               num_heads=num_heads, 
               model_dim=model_dim,
               max_seq_len=max_seq_len).cuda()
    
    # Initialize BF16 for all modules
    model = model.to(torch.bfloat16)
    
    # Apply torch.compile
    compiled_model = torch.compile(model, dynamic=False)
    
    # Store in cache
    _COMPILED_MODEL_CACHE[model_cfg] = compiled_model
    
    return copy.deepcopy(compiled_model)

def fix_state_dict_keys(model_dict, checkpoint_dict):
    """Fix state dict keys to handle compiled model discrepancies.
    
    Args:
        model_dict: The model's state dict
        checkpoint_dict: The checkpoint state dict
        
    Returns:
        Updated checkpoint state dict with correct prefixes
    """
    if not model_dict or not checkpoint_dict:
        return checkpoint_dict
        
    # Get sample keys
    model_keys = list(model_dict.keys())
    checkpoint_keys = list(checkpoint_dict.keys())
    
    if not model_keys or not checkpoint_keys:
        return checkpoint_dict
    
    # Check for _orig_mod prefix
    model_has_prefix = model_keys[0].startswith('_orig_mod.')
    checkpoint_has_prefix = checkpoint_keys[0].startswith('_orig_mod.')
    
    # No changes needed if they match
    if model_has_prefix == checkpoint_has_prefix:
        return checkpoint_dict
    
    # Add or remove prefix as needed
    if model_has_prefix and not checkpoint_has_prefix:
        print0("Adding '_orig_mod.' prefix to checkpoint keys to match compiled model")
        return {f"_orig_mod.{k}": v for k, v in checkpoint_dict.items()}
    elif not model_has_prefix and checkpoint_has_prefix:
        print0("Removing '_orig_mod.' prefix from checkpoint keys to match model")
        return {k.replace('_orig_mod.', ''): v for k, v in checkpoint_dict.items()}
        
    return checkpoint_dict

def compute_minibatch_noise_at_checkpoint(run_id: str, checkpoint_minibatches: List[int], num_minibatches: int, svd_minibatch: int = 4, shared_model=None):
    """Compute minibatch noise at given checkpoints by analyzing gradient variance.
    
    Args:
        run_id: UUID of the training run
        checkpoint_minibatches: List of checkpoint minibatches to analyze (None to process all available)
        num_minibatches: Number of minibatches to analyze per checkpoint
        svd_minibatch: Number of SVD operations to process at once
        shared_model: Reuse an already compiled model
    """
    try:
        # If no checkpoints specified, get all available
        if checkpoint_minibatches is None or len(checkpoint_minibatches) == 0:
            checkpoint_minibatches = get_available_checkpoints(run_id)
            if not checkpoint_minibatches:
                print0(f"No checkpoints found for run_id {run_id}")
                return shared_model  # Return the shared model so it can be reused
            print0(f"Found {len(checkpoint_minibatches)} checkpoints to process: {checkpoint_minibatches}")
        
        # Get hyperparameters for model creation
        args = Hyperparameters()
        
        # Define model configuration
        model_cfg = (args.vocab_size, 16, 8, 1024, max(args.train_seq_len, args.val_seq_len))
        
        # Get or create compiled model (only happens once)
        if shared_model is None:
            baseline_model = get_or_create_compiled_model(model_cfg)
        else:
            baseline_model = shared_model
            print0("Using externally provided model")
        
        # Pre-create batch models once to avoid repeatedly cloning
        num_batch_models = min(8, num_minibatches)  # Limit to reasonable number
        batch_models = [copy.deepcopy(baseline_model) for _ in range(num_batch_models)]
        print0(f"Pre-created {num_batch_models} batch models to reuse")
        
        # Track overall progress
        overall_start_time = time.time()
        checkpoint_times = []
        
        # Process one checkpoint at a time
        for checkpoint_idx, checkpoint_minibatch in enumerate(checkpoint_minibatches):
            try:
                start_time = log_time(f"Starting minibatch noise analysis for checkpoint {checkpoint_minibatch} ({checkpoint_idx+1}/{len(checkpoint_minibatches)})")
                
                # Load checkpoint
                checkpoint_path = f"logs/{run_id}/checkpoint_step{checkpoint_minibatch:06d}.pt"
                if not os.path.exists(checkpoint_path):
                    print0(f"Checkpoint {checkpoint_path} does not exist")
                    continue
                
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                # Setup output directory
                log_dir = Path(f"gradient_logs/{run_id}/{checkpoint_minibatch}")
                if master_process:
                    log_dir.mkdir(parents=True, exist_ok=True)
                
                # Use the original model directly - no need to clone
                model = baseline_model
                
                # Fix state dict mapping for compiled model
                load_start = time.time()
                if 'model' in checkpoint:
                    # Get model state dict for comparison
                    model_state_dict = model.state_dict()
                    
                    # Fix state dict keys using helper function
                    fixed_state_dict = fix_state_dict_keys(model_state_dict, checkpoint['model'])
                    checkpoint['model'] = fixed_state_dict
                
                # Load weights into model
                try:
                    model.load_state_dict(checkpoint["model"], strict=False)
                    load_time = time.time() - load_start
                    print0(f"Model loaded successfully in {load_time:.2f}s")
                    
                    # Update all batch models with the new weights
                    update_start = time.time()
                    for i, batch_model in enumerate(batch_models):
                        batch_model.load_state_dict(checkpoint["model"], strict=False)
                    update_time = time.time() - update_start
                    print0(f"Updated all {len(batch_models)} batch models in {update_time:.2f}s")
                except Exception as e:
                    load_time = time.time() - load_start
                    print(f"Error loading model state dict: {e} (after {load_time:.2f}s)")
                    
                # Distribute model
                for param in model.parameters():
                    dist.broadcast(param.detach(), 0)
                
                # Get data loader
                train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)
                
                # Calculate work distribution across GPUs
                minibatches_per_gpu = num_minibatches // world_size
                remainder = num_minibatches % world_size
                my_minibatch_count = minibatches_per_gpu + (1 if rank < remainder else 0)
                
                # Get filtered parameter matrices
                param_matrices = get_param_matrices(model)
                
                # Initialize gradient storage
                gradients_sum = {key: torch.zeros_like(param, device=device, dtype=torch.bfloat16) 
                               for key, param in param_matrices.items()}
                all_batch_gradients = []
                
                print0(f"This GPU will process {my_minibatch_count} minibatches serially")
                
                # ======== 1. GRADIENT COMPUTATION PHASE ========
                # Process minibatches one at a time
                batch_load_time = 0
                forward_time = 0
                backward_time = 0
                extract_time = 0
                model_clone_time = 0
                
                for batch_idx in range(my_minibatch_count):
                    # Get a pre-created batch model and reset it
                    batch_model_idx = batch_idx % num_batch_models
                    batch_model = batch_models[batch_model_idx]
                    batch_model.train()
                    batch_model.zero_grad(set_to_none=True)
                    
                    try:
                        # Data loading
                        data_start = time.time()
                        inputs, targets = next(train_loader)
                        window_size_blocks = get_window_size_blocks(checkpoint_minibatch)
                        data_end = time.time()
                        batch_load_time += (data_end - data_start)
                        
                        # Forward pass
                        forward_start = time.time()
                        loss = batch_model(inputs, targets, window_size_blocks)
                        torch.cuda.synchronize()
                        forward_end = time.time()
                        forward_time += (forward_end - forward_start)
                        
                        # Backward pass
                        backward_start = time.time()
                        loss.backward()
                        torch.cuda.synchronize()
                        backward_end = time.time()
                        backward_time += (backward_end - backward_start)
                        
                        # Extract gradients
                        extract_start = time.time()
                        batch_gradients = {}
                        for key, param in param_matrices.items():
                            for name, bparam in batch_model.named_parameters():
                                if name == key[1] and bparam.grad is not None:
                                    grad = bparam.grad.clone().to(torch.bfloat16)
                                    batch_gradients[key] = grad
                                    
                                    # Average gradients across GPUs
                                    dist.all_reduce(grad, op=dist.ReduceOp.AVG)
                                    gradients_sum[key] += grad
                                    break
                        
                        all_batch_gradients.append(batch_gradients)
                        extract_end = time.time()
                        extract_time += (extract_end - extract_start)
                        
                        print0(f"Completed minibatch {batch_idx+1}/{my_minibatch_count}")
                        
                    except Exception as e:
                        print(f"Error processing batch {batch_idx}: {e}")
                
                # Report timing breakdown
                if master_process:
                    print(f"Gradient Computation Timing for Checkpoint {checkpoint_minibatch}:")
                    print(f"  - Data loading: {batch_load_time:.2f}s")
                    print(f"  - Forward pass: {forward_time:.2f}s")
                    print(f"  - Backward pass: {backward_time:.2f}s")
                    print(f"  - Gradient extraction: {extract_time:.2f}s")
                    print(f"  - Total: {batch_load_time + forward_time + backward_time + extract_time:.2f}s")
                
                # Final reduction of gradient sums
                for key, grad_sum in gradients_sum.items():
                    dist.all_reduce(grad_sum, op=dist.ReduceOp.SUM)
                
                # Calculate average gradients
                gradients_avg = {key: grad_sum / num_minibatches for key, grad_sum in gradients_sum.items()}
                
                # Memory cleanup - don't delete model yet as we need to release it outside the loop
                torch.cuda.empty_cache()
                gradient_time = log_time("Gradient computation completed", start_time)
                
                # ======== 2. SVD AND ANALYSIS PHASE ========
                svd_time = log_time("Starting SVD computation")
                
                # Split QKV components
                split_gradients_avg = split_qkv_tensors(gradients_avg)
                split_batch_gradients = [split_qkv_tensors(batch_grads) for batch_grads in all_batch_gradients]
                
                # Gather batch gradients from all processes
                all_split_batch_gradients = [None] * world_size
                all_split_batch_gradients[rank] = split_batch_gradients
                
                dist.barrier()
                dist.all_gather_object(all_split_batch_gradients, split_batch_gradients)
                
                # Combine all minibatches from all ranks
                combined_split_batch_gradients = []
                for process_gradients in all_split_batch_gradients:
                    if process_gradients:
                        for batch_grads in process_gradients:
                            device_grads = {k: (g.to(device) if g is not None else None) for k, g in batch_grads.items()}
                            combined_split_batch_gradients.append(device_grads)
                
                del all_split_batch_gradients
                torch.cuda.empty_cache()
                
                print0(f"Combined all split gradients. Total minibatches: {len(combined_split_batch_gradients)}")
                
                # Calculate Frobenius norms for all keys
                all_keys = list(split_gradients_avg.keys())
                frobenius_norms = {key: torch.norm(grad).item() for key, grad in split_gradients_avg.items() if grad is not None}
                
                # Compute SVDs for average gradients for all keys
                avg_svd_results = {}
                
                # Process average gradient SVD in batches
                for i in range(0, len(all_keys), svd_minibatch):
                    batch_keys = all_keys[i:i+svd_minibatch]
                    
                    if master_process and i % (svd_minibatch * 5) == 0:
                        print(f"Processing average SVD batch {i//svd_minibatch + 1}/{len(all_keys)//svd_minibatch + 1}")
                    
                    for key in batch_keys:
                        if key not in split_gradients_avg or split_gradients_avg[key] is None or split_gradients_avg[key].ndim <= 1:
                            continue
                            
                        try:
                            grad = split_gradients_avg[key]
                            # Convert to float32 for SVD operation
                            grad_f32 = grad.to(torch.float32)
                            # Perform SVD in float32
                            U, S, Vh = torch.linalg.svd(grad_f32, full_matrices=False)
                            # Convert results back to bfloat16 for storage
                            avg_svd_results[key] = (U.to(torch.bfloat16), S.to(torch.bfloat16), Vh.to(torch.bfloat16))
                            
                            if master_process:
                                log_dir = Path(f"gradient_logs/{run_id}/{checkpoint_minibatch}")
                                torch.save(grad, log_dir / f"layer_{key[0]}_weight_{key[1].replace('.', '_')}_avg.pt")
                                
                        except Exception as e:
                            print(f"SVD failed for key {key}: {e}")
                    
                    # Clear cache after each batch
                    torch.cuda.empty_cache()
                
                log_time("Average gradient SVD completed")
                
                # NEW APPROACH: Divide minibatches evenly among ranks
                # Calculate how many minibatches each rank should process
                total_minibatches = len(combined_split_batch_gradients)
                minibatches_per_rank = total_minibatches // world_size
                remainder = total_minibatches % world_size
                
                # Each rank gets minibatches_per_rank or minibatches_per_rank+1 minibatches
                my_start_idx = rank * minibatches_per_rank + min(rank, remainder)
                my_end_idx = my_start_idx + minibatches_per_rank + (1 if rank < remainder else 0)
                my_minibatches = list(range(my_start_idx, my_end_idx))
                
                print0(f"Work distribution: {world_size} ranks processing {total_minibatches} minibatches")
                print(f"Rank {rank} will process minibatches {my_minibatches}")
                
                # Process per-minibatch SVDs and collect singular values
                singular_values_by_key = {}
                
                # Each rank processes its assigned minibatches
                for batch_idx in my_minibatches:
                    batch_grads = combined_split_batch_gradients[batch_idx]
                    batch_start_time = time.time()
                    
                    # For each minibatch, process all keys in batches
                    for i in range(0, len(all_keys), svd_minibatch):
                        batch_keys = all_keys[i:i+svd_minibatch]
                        
                        for key in batch_keys:
                            if (key not in avg_svd_results or key not in batch_grads or 
                                batch_grads[key] is None or batch_grads[key].ndim <= 1):
                                continue
                                
                            try:
                                # Align gradient with average
                                grad = batch_grads[key]
                                avg_grad = split_gradients_avg[key]
                                aligned_grad = procrustes_align_gpu(grad, avg_grad)
                                
                                # Get singular values
                                _, S_i, _ = torch.linalg.svd(aligned_grad.to(torch.float32), full_matrices=False)
                                S_avg = avg_svd_results[key][1]
                                
                                # Record singular values
                                formatted_name = format_weight_name(key[1])
                                frob_norm = frobenius_norms.get(key, 0.0)
                                
                                min_size = min(len(S_i), len(S_avg))
                                for idx in range(min_size):
                                    dict_key = (key[0], formatted_name, idx)
                                    
                                    if dict_key not in singular_values_by_key:
                                        singular_values_by_key[dict_key] = {
                                            'layer_number': key[0],
                                            'weight_name': formatted_name,
                                            'singular_value_index': idx,
                                            'G_avg_singular_value': S_avg[idx].item(),
                                            'G_i_singular_value_list': [],
                                            'frobenius_norm': frob_norm,
                                            'checkpoint_minibatch': checkpoint_minibatch
                                        }
                                    
                                    singular_values_by_key[dict_key]['G_i_singular_value_list'].append(S_i[idx].item())
                                    
                            except Exception as e:
                                print(f"Error processing batch {batch_idx}, key {key}: {e}")
                        
                        # Clear cache frequently
                        torch.cuda.empty_cache()
                    
                    batch_end_time = time.time()
                    # print(f"Rank {rank}: Processed minibatch {batch_idx} in {batch_end_time - batch_start_time:.2f}s")
                
                # Gather results from all ranks
                my_results = list(singular_values_by_key.values())
                results_list = [None] * world_size
                results_list[rank] = my_results
                
                dist.barrier()
                dist.all_gather_object(results_list, my_results)
                
                # Combine results and compute statistics
                combined_results_dict = {}
                for rank_results in results_list:
                    if rank_results:
                        for entry in rank_results:
                            dict_key = (entry['layer_number'], entry['weight_name'], entry['singular_value_index'])
                            
                            if dict_key not in combined_results_dict:
                                combined_results_dict[dict_key] = entry.copy()
                            else:
                                combined_results_dict[dict_key]['G_i_singular_value_list'].extend(
                                    entry['G_i_singular_value_list']
                                )
                
                # Compute statistics
                final_results = []
                for entry in combined_results_dict.values():
                    sv_array = np.array(entry['G_i_singular_value_list'])
                    final_results.append({
                        'layer_number': entry['layer_number'],
                        'weight_name': entry['weight_name'],
                        'singular_value_index': entry['singular_value_index'],
                        'G_avg_singular_value': entry['G_avg_singular_value'],
                        'singular_value_variance': float(np.var(sv_array)),
                        'singular_value_mean': float(np.mean(sv_array)),
                        'count': len(sv_array),
                        'G_i_singular_value_list': entry['G_i_singular_value_list'],
                        'frobenius_norm': entry['frobenius_norm'],
                        'checkpoint_minibatch': entry['checkpoint_minibatch']
                    })
                    
                dataframe_time = log_time("SVD and Procrustes alignment completed", svd_time)
                
                # Save results
                if master_process and final_results:
                    df = pd.DataFrame(final_results)
                    if not df.empty:
                        csv_df = df.drop(columns=['G_i_singular_value_list'])
                        csv_df.to_csv(log_dir / "minibatch_noise.csv", index=False)
                        df.to_pickle(log_dir / "minibatch_noise_with_lists.pkl")
                        print0(f"Saved minibatch noise analysis to {log_dir}/minibatch_noise.csv")
                
                # Report timing
                final_time = log_time("DataFrame processing completed", dataframe_time)
                total_time = final_time - start_time
                
                if master_process:
                    print(f"Rank {rank}: Timing breakdown for checkpoint {checkpoint_minibatch}:")
                    print(f"  - Gradient computation:   {gradient_time - start_time:.2f}s ({(gradient_time-start_time)/total_time*100:.1f}%)")
                    print(f"  - SVD and Procrustes:     {dataframe_time - svd_time:.2f}s ({(dataframe_time-svd_time)/total_time*100:.1f}%)")
                    print(f"  - DataFrame processing:   {final_time - dataframe_time:.2f}s ({(final_time-dataframe_time)/total_time*100:.1f}%)")
                    print(f"  - Total execution time:   {total_time:.2f}s (100%)")
                
                # Clean up all variables to avoid memory leaks
                cleanup_vars = [checkpoint, train_loader, gradients_sum, all_batch_gradients,
                             param_matrices, gradients_avg, split_gradients_avg, split_batch_gradients,
                             combined_split_batch_gradients, avg_svd_results,
                             singular_values_by_key, my_results, results_list, combined_results_dict,
                             final_results, frobenius_norms]
                
                for var in cleanup_vars:
                    del var
                
                # Do NOT delete model - it's being reused
                
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                print0(f"Finished processing checkpoint {checkpoint_minibatch} ({checkpoint_idx+1}/{len(checkpoint_minibatches)}), memory cleaned up.")
                
                # Track checkpoint time for progress estimation
                checkpoint_times.append(time.time() - start_time)
                avg_time_per_checkpoint = sum(checkpoint_times) / len(checkpoint_times)
                remaining_checkpoints = len(checkpoint_minibatches) - (checkpoint_idx + 1)
                est_remaining_time = remaining_checkpoints * avg_time_per_checkpoint
                
                # Show progress and time estimates
                if master_process and remaining_checkpoints > 0:
                    print(f"\nPROGRESS UPDATE:")
                    print(f"  - Checkpoints completed: {checkpoint_idx+1}/{len(checkpoint_minibatches)} ({(checkpoint_idx+1)/len(checkpoint_minibatches)*100:.1f}%)")
                    print(f"  - Average time per checkpoint: {avg_time_per_checkpoint:.1f}s")
                    print(f"  - Estimated time remaining: {est_remaining_time/60:.1f} minutes ({est_remaining_time/3600:.1f} hours)")
                    print(f"  - Elapsed time: {(time.time() - overall_start_time)/60:.1f} minutes\n")
                
            except Exception as e:
                print(f"Error processing checkpoint {checkpoint_minibatch}: {e}")
                import traceback
                traceback.print_exc()
                
                # Clean up any remaining objects
                gc.collect()
                torch.cuda.empty_cache()
        
        # All checkpoints have been processed
        total_time = time.time() - overall_start_time
        if master_process:
            print("\n" + "="*50)
            print(f"PROCESSING COMPLETE FOR RUN {run_id}")
            print(f"  - Processed {len(checkpoint_minibatches)} checkpoints in {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
            if checkpoint_times:
                print(f"  - Average time per checkpoint: {sum(checkpoint_times)/len(checkpoint_times):.1f} seconds")
                print(f"  - Fastest checkpoint: {min(checkpoint_times):.1f} seconds")
                print(f"  - Slowest checkpoint: {max(checkpoint_times):.1f} seconds")
            print("="*50 + "\n")
            
        print0(f"Completed processing all {len(checkpoint_minibatches)} checkpoints for run_id {run_id}")
        
        # Return the shared model for reuse
        return baseline_model
    
    finally:
        # We don't destroy the process group as it would be reused across checkpoints
        pass

def compute_minibatch_noise_for_multiple_runs(run_ids: List[str], checkpoint_minibatches_dict=None, num_minibatches=32, svd_minibatch=4):
    """
    Process multiple runs, reusing the compiled model across all checkpoints.
    
    Args:
        run_ids: List of run IDs to process
        checkpoint_minibatches_dict: Dictionary mapping run_id to list of checkpoints to analyze
        num_minibatches: Number of minibatches to analyze per checkpoint
        svd_minibatch: Number of SVD operations to process at once
    """
    if checkpoint_minibatches_dict is None:
        checkpoint_minibatches_dict = {}
    
    # Initialize a shared model that will be reused across all checkpoints
    shared_model = None
    
    try:
        for run_id in run_ids:
            # Get checkpoints from dict if specified, otherwise None to process all available
            checkpoints = checkpoint_minibatches_dict.get(run_id, None)
            print0(f"Processing run_id: {run_id}")
            shared_model = compute_minibatch_noise_at_checkpoint(
                run_id=run_id,
                checkpoint_minibatches=checkpoints,
                num_minibatches=num_minibatches,
                svd_minibatch=svd_minibatch,
                shared_model=shared_model
            )
    finally:
        # Ensure process group is destroyed after all processing
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True, help="UUID of the training run")
    parser.add_argument("--checkpoint_minibatches", type=int, nargs='+', default=None, 
                      help="List of checkpoint minibatches to analyze (default: all available)")
    parser.add_argument("--num_minibatches", type=int, default=32, help="Number of minibatches to analyze")
    parser.add_argument("--svd_minibatch", type=int, default=4, 
                      help="Number of SVD operations to process at once (default: 4)")
    args_cli = parser.parse_args()
    
    # Use the multi-run handler for better efficiency
    compute_minibatch_noise_for_multiple_runs(
        run_ids=[args_cli.run_id],
        checkpoint_minibatches_dict={args_cli.run_id: args_cli.checkpoint_minibatches} if args_cli.checkpoint_minibatches else {},
        num_minibatches=args_cli.num_minibatches,
        svd_minibatch=args_cli.svd_minibatch
    )