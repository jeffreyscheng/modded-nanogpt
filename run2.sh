#!/bin/bash
# 0.1	0.003	0.03	0.01	
warmdown_iters_list=(600 926 1200 1500 1800)
newton_schulz_iters_list=(10 20 30)


# Loop through all combinations
for warmdown_iters in "${warmdown_iters_list[@]}"; do
    for newton_schulz_iter in "${newton_schulz_iters_list[@]}"; do
        echo "Running with warmdown_iters=${warmdown_iters}, newton_schulz_iter=${newton_schulz_iter}"
        torchrun --standalone --nproc_per_node=8 train_gpt2.py \
            --num_iterations 3300 \
            --embedding_lr 0.1 \
            --lm_head_lr 0.003 \
            --muon_lr 0.03 \
            --scalar_lr 0.01 \
            --muon_momentum 0.95 \
            --warmdown_iters ${warmdown_iters} \
            --newton_schulz_iters ${newton_schulz_iter}
    done
done
