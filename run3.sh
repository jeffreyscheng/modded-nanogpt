#!/bin/bash
# stiefel_descent_lr_list=(0.03)
stiefel_descent_lr_list=(0.1 0.03 0.01 0.003 0.001)
# Loop through all combinations
for stiefel_descent_lr in "${stiefel_descent_lr_list[@]}"; do
    echo "Running with steifel_descent_lr=${stiefel_descent_lr}"
    torchrun --standalone --nproc_per_node=2 train_gpt2.py \
        --num_iterations 3300 \
        --muon_lr ${stiefel_descent_lr} \
        --use_stiefel_descent=True \
        --val_loss_every 100 \
        --warmdown_iters 926 \
        --newton_schulz_iters 5 \
        --orthogonalize_every 10
done
