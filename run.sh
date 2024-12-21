#!/bin/bash
for lr in 1e-2 3e-3 1e-3 3e-4 1e-4 3e-5 1e-5; do
    for cooldown in 400 800 1200; do
        for warmup in 0 200; do
            echo "Running with lr=$lr, cooldown=$cooldown, warmup=$warmup"
            torchrun --standalone --nproc_per_node=1 train_gpt2.py \
                --muon_lr $lr \
                --cooldown_iters $cooldown \
                --warmup_iters $warmup
        done
    done
done