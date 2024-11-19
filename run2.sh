#!/bin/bash
# 0.1	0.003	0.03	0.01
muon_lr_list=(0.028 0.03 0.032 0.034 0.036 0.038 0.04)	
warmdown_iters_list=(600 926 1200 1500 1800 2100 2400)
newton_schulz_iters=4


# Loop through all combinations
for warmdown_iters in "${warmdown_iters_list[@]}"; do
    for muon_lr in "${muon_lr_list[@]}"; do
        echo "Running with warmdown_iters=${warmdown_iters}, muon_lr=${muon_lr}"
        torchrun --standalone --nproc_per_node=2 train_gpt2.py \
            --num_iterations 3300 \
            --embedding_lr 0.1 \
            --lm_head_lr 0.003 \
            --muon_lr ${muon_lr} \
            --scalar_lr 0.01 \
            --muon_momentum 0.95 \
            --warmdown_iters ${warmdown_iters} \
            --newton_schulz_iters ${newton_schulz_iters}
    done
done
