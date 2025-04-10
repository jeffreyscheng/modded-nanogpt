#!/usr/bin/env python3
# Simple script to find optimal bfloat16 quantization of polynomial coefficients

import torch
import torch.distributed as dist
import tqdm
import struct
import os
import time
import argparse
import sys

# Initialize distributed training if available
def init_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        
        # Initialize process group
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        print(f"Distributed initialization: rank {rank}/{world_size} on GPU {local_rank}")
        return rank, world_size
    else:
        print("Running in non-distributed mode")
        return 0, 1

# ======= PASTE YOUR COEFFICIENTS HERE =======
coefficients = [
    [4.510918617248535, -7.218330383300781, 2.7313356399536133],
    [4.605247974395752, -6.573604106903076, 2.3541903495788574],
    [4.919572353363037, -5.963148593902588, 1.814536452293396],
    [4.132555961608887, -3.550161123275757, 0.7916041016578674],
    [3.9480252265930176, -3.3081469535827637, 0.7301044464111328],
    [2.495997905731201, -1.7606257200241089, 0.3820432126522064],
    [2.0578646659851074, -1.5518815517425537, 0.48931562900543213],
]
# ============================================

# Helper functions for float to bfloat16 conversion
def float_to_bits(f):
    """Convert float to its binary representation as uint32"""
    return struct.unpack('I', struct.pack('f', f))[0]

def float_to_bfloat16_bits(f):
    """Convert float to bfloat16 bits (uint16)"""
    return float_to_bits(f) >> 16

def bits_to_binary_string(bits, bits_count=16):
    """Convert uint32 to binary string representation"""
    return format(bits, f'0{bits_count}b')

def get_exact_bfloat16_value(f):
    """Get exact bfloat16 representation of a float as a float"""
    bf16_bits = float_to_bfloat16_bits(f)
    bf16_as_f32_bits = bf16_bits << 16
    return struct.unpack('f', struct.pack('I', bf16_as_f32_bits))[0]

def print_exact_bfloat16_coefficients(coeffs, title="Coefficients", should_print=True):
    """Print exact bfloat16 binary representation of coefficients"""
    if not should_print:
        return
        
    print(f"\n{title} (exact bfloat16 decimal values):")
    for i, layer in enumerate(coeffs):
        exact_values = [get_exact_bfloat16_value(x) for x in layer]
        print(f"    [{exact_values[0]}, {exact_values[1]}, {exact_values[2]}],")
    
    print(f"\n{title} (as binary representation):")
    for i, layer in enumerate(coeffs):
        binary_values = [bits_to_binary_string(float_to_bfloat16_bits(x)) for x in layer]
        print(f"[{binary_values[0]}, {binary_values[1]}, {binary_values[2]}],")
    
    print(f"\n{title} (polynomial representation for Desmos):")
    for i, layer in enumerate(coeffs):
        exact_values = [get_exact_bfloat16_value(x) for x in layer]
        print(f"f_{i}(x) = {exact_values[0]}x + {exact_values[1]}x^3 + {exact_values[2]}x^5")

# Evaluation functions
def evaluate_polynomial(coeffs, x):
    result = x
    for poly_coeffs in coeffs:
        result = sum(c * torch.pow(result, 2*i+1) for i, c in enumerate(poly_coeffs))
    return result

def univariate_loss(coeffs, device):
    x = torch.linspace(0.0, 1.0, 100000).to(device)
    y = evaluate_polynomial(coeffs, x)
    return torch.norm(y - 1).item() / 100000

def round_up_to_bfloat16(x):
    int_repr = x.view(torch.int32)
    mask = (1 << 16) - 1
    has_lower_bits = (int_repr & mask) != 0
    round_up = torch.where(has_lower_bits, torch.tensor(1 << 16, dtype=torch.int32), torch.tensor(0, dtype=torch.int32))
    return (int_repr + round_up & ~mask).view(torch.float32)

def round_down_to_bfloat16(x):
    int_repr = x.view(torch.int32)
    mask = ~((1 << 16) - 1)
    return (int_repr & mask).view(torch.float32)

def main():
    # Initialize distributed training
    rank, world_size = init_distributed()
    master_process = (rank == 0)
    
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    print(f"Rank {rank} using device: {device}")
    
    # Allocate CUDA memory early
    warmup_tensor = torch.zeros(10, device=device)
    del warmup_tensor

    # Flatten coefficients for processing
    flat_coeffs = []
    shape = []
    for layer in coefficients:
        shape.append(len(layer))
        flat_coeffs.extend(layer)

    tensor_coeffs = torch.tensor(flat_coeffs, dtype=torch.float32, device=device)
    n = len(flat_coeffs)
    combinations = 2**n
    
    # Evaluate original coefficients
    original_loss = univariate_loss(coefficients, device)
    if master_process:
        print(f"Original loss: {original_loss:.10f}")
        print_exact_bfloat16_coefficients(coefficients, "Original coefficients")
        print(f"\nTesting {combinations} combinations for {n} coefficients across {world_size} GPUs...")
    
    # Divide work among available GPUs
    chunk_size = (combinations + world_size - 1) // world_size
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, combinations)
    
    # Record start time
    torch.cuda.synchronize()
    start_time = time.time()
    
    # Variables to track best result for this process
    best_loss = float('inf')
    best_coeffs = None
    
    # Process combinations assigned to this GPU
    for i in tqdm.tqdm(range(start_idx, end_idx), disable=not master_process):
        binary = format(i, f'0{n}b')
        new_coeffs = torch.zeros_like(tensor_coeffs)
        
        for j in range(n):
            if binary[j] == '1':
                new_coeffs[j] = round_up_to_bfloat16(tensor_coeffs[j:j+1])
            else:
                new_coeffs[j] = round_down_to_bfloat16(tensor_coeffs[j:j+1])
        
        # Reshape to original structure
        structured_coeffs = []
        idx = 0
        for size in shape:
            structured_coeffs.append(new_coeffs[idx:idx+size].tolist())
            idx += size
        
        loss = univariate_loss(structured_coeffs, device)
        
        if loss < best_loss:
            best_loss = loss
            best_coeffs = structured_coeffs
            if master_process or world_size == 1:
                print(f"Rank {rank}: New best at {i+1}/{combinations}: loss = {loss:.10f}")
                print_exact_bfloat16_coefficients(structured_coeffs, f"New best coefficients (rank {rank})")
    
    # Gather results from all processes
    if world_size > 1:
        # Create tensors to hold results from all processes
        all_losses = torch.tensor([best_loss], device=device)
        gathered_losses = [torch.zeros(1, device=device) for _ in range(world_size)]
        
        # Gather losses from all processes
        dist.all_gather(gathered_losses, all_losses)
        
        # Find the global best loss and its rank
        global_best_loss = float('inf')
        global_best_rank = -1
        
        for r, loss_tensor in enumerate(gathered_losses):
            loss = loss_tensor.item()
            if loss < global_best_loss:
                global_best_loss = loss
                global_best_rank = r
        
        # Have the process with the best loss broadcast its coefficients
        is_best_process = (rank == global_best_rank)
        
        if is_best_process:
            best_coeffs_flat = torch.tensor(flat_coeffs, device=device)  # Placeholder for size
            idx = 0
            for layer in best_coeffs:
                for coeff in layer:
                    best_coeffs_flat[idx] = coeff
                    idx += 1
        else:
            best_coeffs_flat = torch.zeros_like(tensor_coeffs)
        
        # Broadcast the best coefficients
        dist.broadcast(best_coeffs_flat, global_best_rank)
        
        # Reconstruct the coefficients
        if not is_best_process:
            best_coeffs = []
            idx = 0
            for size in shape:
                best_coeffs.append(best_coeffs_flat[idx:idx+size].tolist())
                idx += size
        
        # Update best_loss with the global best
        best_loss = global_best_loss

    # Print final results on master process
    if master_process:
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        
        print(f"\nSearch completed in {elapsed_time:.2f} seconds")
        print(f"Best loss: {best_loss:.10f} (improved by {(original_loss - best_loss) / original_loss * 100:.2f}%)")
        print("\nFinal best coefficients:")
        print_exact_bfloat16_coefficients(best_coeffs, "Final best coefficients")
    
    # Clean up
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main() 