#!/bin/bash

# Set OpenMP threads explicitly to silence the warning
export OMP_NUM_THREADS=1

# Run distributed training
torchrun --standalone --nproc_per_node=8 train_polynomial.py 