"""
train_gptm_armijo.py
====================
Back‑tracking Armijo search that chooses a step size **alpha** along the joint
*optimiser update direction* D so that the **validation loss** is minimised.

This version fixes the checkpoint‑loading mismatch that you spotted:

* Training writes `optimizers=[opt1_state, opt2_state]` (a *list*).  We now
  recreate **both** optimisers (AdamW + Muon) in the same order and load that
  list correctly.
* The search therefore uses the *exact* combined update of both optimisers
  when computing D.

Memory rules remain unchanged: only one full copy of the model lives on GPU at
any time; direction and reference weights are kept on CPU.
"""

from __future__ import annotations

import os
import gc
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from gpt_static import (
    Hyperparameters,
    GPT,
    Muon,  # custom optimiser defined in gpt_static
    run_validation,
    load_state_dict_safely,
    distributed_data_generator,
    get_window_size_blocks,
)

import torch, gc, inspect, os, sys
from collections import defaultdict

import os, torch, torch.distributed as dist

# --- distributed init (env:// works with torchrun) --------------------------
if not dist.is_initialized():
    dist.init_process_group(backend="nccl", init_method="env://")

rank        = dist.get_rank()         # 0‑‑7
world_size  = dist.get_world_size()
local_rank  = int(os.environ.get("LOCAL_RANK", rank))   # 0‑‑7 on a single node

torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

if rank == 0:
    print(f"rank {rank}/{world_size} using {device}")

# -----------------------------------------------------------------------------
# Armijo hyper‑parameters
# -----------------------------------------------------------------------------
starting_alpha = 4.0
shrink = 0.5          # alpha <- alpha * shrink each back‑track step
min_alpha = 1e-5
armijo_c1 = 1e-4      # sufficient‑decrease constant

# -----------------------------------------------------------------------------
# Model helpers (copied from make_soup_picture.py style)
# -----------------------------------------------------------------------------

def _build_model(device: str) -> GPT:
    """Create and compile the same GPT‑M architecture used during training."""
    hp = Hyperparameters()
    model = GPT(
        vocab_size=hp.vocab_size,
        num_layers=16,
        num_heads=8,
        model_dim=1024,
        max_seq_len=max(hp.train_seq_len, hp.val_seq_len),
    ).to(device)

    # cast embeddings to bf16 (as done in make_soup_picture.py)
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            m.weight.data = m.weight.data.to(torch.bfloat16)
    for p in model.parameters():
        if p.is_floating_point() and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)

    model = torch.compile(model, dynamic=False)
    return model


# -----------------------------------------------------------------------------
# Optimiser helpers
# -----------------------------------------------------------------------------

def _build_optimizers(
    model: GPT,
    rank: int,
    world_size: int,
    optimizer_states: List[dict] | None,
    global_step: int,
) -> List[torch.optim.Optimizer]:
    hidden_matrix_params = sorted((p for p in model.blocks.parameters() if p.ndim >= 2), key=lambda x: x.size(), reverse=True)
    embed_params = [*model.embed.parameters(), *model.value_embeds.parameters()]
    scalar_params = [model.scalars]
    head_params: list[nn.Parameter] = [model.lm_head_w]
    # sanity check
    params_collections = [hidden_matrix_params, embed_params, scalar_params, head_params]
    optimized_parameters_set = {p for params in params_collections for p in params}
    assert optimized_parameters_set == {*model.parameters()}
    assert len(optimized_parameters_set) == sum(len(lst) for lst in params_collections)

    # init the optimizer(s)
    adam_param_groups = [dict(params=head_params, lr=1/320), dict(params=embed_params, lr=0.3), dict(params=scalar_params, lr=0.015)]

    opt1 = torch.optim.AdamW(
        adam_param_groups,
        betas=(0.8, 0.95),
        eps=1e-10,
        weight_decay=0.0,
        fused=True,
    )
    opt2 = Muon(
        hidden_matrix_params,
        lr=0.025,
        momentum=0.95,
        rank=rank,
        world_size=world_size,
    )

    optimizers: List[torch.optim.Optimizer] = [opt1, opt2]

    if optimizer_states:                       # only rank 0 has them
        assert len(optimizer_states) == len(optimizers)
        for opt, state in zip(optimizers, optimizer_states):
            opt.load_state_dict(state)
            for p,st in opt2.state.items(): st["mantissa"] = st.get("mantissa", torch.zeros_like(p, dtype=torch.uint16)).to(p.device, torch.uint16)
            for g in opt.param_groups:
                g["step"] = global_step

    return optimizers


# -----------------------------------------------------------------------------
# Compute descent direction D
# -----------------------------------------------------------------------------

def _compute_direction(
    model: GPT,
    optimizers: List[torch.optim.Optimizer],
    step: int,
    rank: int,
    world_size: int,
    seed: int,
) -> List[torch.Tensor]:
    """Return the optimiser *update* direction D (CPU bf16 tensors)."""
    # Save starting weights on CPU
    params_0_cpu = [p.detach().cpu().clone().to(torch.bfloat16) for p in model.parameters()]

    # ---------------- perform one simulated training step (rank 0 only) ----
    model.train()
    hp = Hyperparameters()
    batch_size = world_size * hp.train_seq_len
    train_iter = distributed_data_generator(
        hp.train_files,
        batch_size,
        rank,
        world_size,
        seed=seed,
    )
    inp, tgt = next(train_iter)
    for opt in optimizers:
        opt.zero_grad(set_to_none=True)
    loss = model(inp, tgt, get_window_size_blocks(step))
    loss.backward()
    for opt in optimizers:
        opt.step()
    del inp, tgt, loss
    torch.cuda.empty_cache()

    # ---------------- build CPU direction list -----------------------------
    direction_cpu: List[torch.Tensor] = []
    for p, p0 in zip(model.parameters(), params_0_cpu):
        diff_cpu = (p.detach().cpu().to(torch.bfloat16) - p0)
        direction_cpu.append(diff_cpu)

    # restore original weights in‑place
    for p, p0 in zip(model.parameters(), params_0_cpu):
        p.data.copy_(p0.to(p.device), non_blocking=True)

    # broadcast D to all ranks (CPU tensors)
    if world_size > 1:
        # rank 0 owns the real tensors; others allocate tmp buffers on GPU
        direction_cuda = (
            [d.to(device, non_blocking=True) for d in direction_cpu]
            if rank == 0
            else [torch.empty_like(p, dtype=torch.bfloat16, device=device)
                  for p in model.parameters()]
        )
        for d in direction_cuda:
            dist.broadcast(d, src=0)          # NCCL can handle this
        # non‑0 ranks: bring data back to CPU so D lives off‑device
        if rank != 0:
            direction_cpu = [d.cpu() for d in direction_cuda]
        # free the CUDA buffers on every rank
        del direction_cuda
    gc.collect(); torch.cuda.empty_cache()
    return direction_cpu


# -----------------------------------------------------------------------------
# Apply / rollback helpers (stream CPU -> GPU slice by slice)
# -----------------------------------------------------------------------------

def _apply_direction(model: GPT, direction: List[torch.Tensor], alpha: float):
    """In‑place: theta <- theta + alpha * D (slice streaming)."""
    for p, d_cpu in zip(model.parameters(), direction):
        d_gpu = d_cpu.to(p.device, dtype=p.dtype, non_blocking=True)
        p.data.add_(alpha, d_gpu)
        del d_gpu
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Validation evaluation helpers
# -----------------------------------------------------------------------------

def evaluate_one_armijo_alpha(
    alpha: float,
    model: GPT,
    rank: int,
    device: str,
    world_size: int,
) -> torch.Tensor:
    direction = model._direction  # type: ignore[attr-defined]
    _apply_direction(model, direction, alpha)
    loss = run_validation(
        model,
        model._step,  # type: ignore[attr-defined]
        Hyperparameters(),
        rank,
        world_size,
        seed=1234 + int(alpha * 1000),
    ).detach()
    _apply_direction(model, direction, -alpha)  # rollback
    return loss


def passes_armijo_condition(loss_alpha: torch.Tensor, alpha: float, model: GPT) -> bool:
    f0 = model._f0  # type: ignore[attr-defined]
    g0 = model._g0  # type: ignore[attr-defined]
    return bool(loss_alpha <= f0 + armijo_c1 * alpha * g0)


# -----------------------------------------------------------------------------
# Main search driver
# -----------------------------------------------------------------------------

def armijo_search(
    checkpoint_path: str,
    rank: int,
    device: str,
    world_size: int,
) -> Tuple[float, float]:
    """Run Armijo search. Return (best_alpha, best_val_loss)."""

    # ------------- rank 0 loads checkpoint meta on CPU --------------------
    if rank == 0:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        model_state = ckpt.get("model")
        optim_states = ckpt.get("optimizers", [])
        step = ckpt.get("step", 0)
    else:
        model_state, optim_states, step = None, [], 0

    # broadcast step so everyone knows seed, etc.
    step_t = torch.tensor([step], dtype=torch.long, device=device)
    dist.broadcast(step_t, 0)
    step = int(step_t.item())

    # ------------- build model and load weights ---------------------------
    model = _build_model(device)
    model._step = step  # type: ignore[attr-defined]

    if rank == 0 and model_state is not None:
        load_state_dict_safely(model, model_state, strict=False)

    # broadcast parameters to all ranks
    for p in model.parameters():
        dist.broadcast(p.data, 0)

    # ------------- recreate optimisers & compute D ------------------------
    optimizers = _build_optimizers(model, rank, world_size, optim_states, step)
    direction = _compute_direction(model, optimizers, step, rank, world_size, seed=step + 1)
    model._direction = direction  # type: ignore[attr-defined]

    # ------------- baseline val loss + directional derivative -------------
    model.eval()
    f0 = run_validation(model, step, Hyperparameters(), rank, world_size, seed=step + 2).detach()
    eps = 1e-4
    _apply_direction(model, direction, eps)
    f_eps = run_validation(model, step, Hyperparameters(), rank, world_size, seed=step + 3).detach()
    _apply_direction(model, direction, -eps)
    g0 = (f_eps - f0) / eps

    model._f0 = f0  # type: ignore[attr-defined]
    model._g0 = g0  # type: ignore[attr-defined]

    if rank == 0:
        print(f"[Armijo] baseline val_loss={f0.item():.6f}, g0≈{g0.item():.6e}")

    # ------------- back‑tracking loop ------------------------------------
    alpha = starting_alpha
    best_alpha, best_loss = 0.0, f0

    while alpha >= min_alpha:
        loss_alpha = evaluate_one_armijo_alpha(alpha, model, rank, device, world_size)
        if rank == 0:
            print(f"[Armijo] alpha={alpha:.6f} -> val_loss={loss_alpha.item():.6f}")

        if loss_alpha < best_loss:
            best_alpha, best_loss = alpha, loss_alpha

        if passes_armijo_condition(loss_alpha, alpha, model):
            if rank == 0:
                print("[Armijo] Armijo condition satisfied — stopping.")
            break

        alpha *= shrink

    return best_alpha, best_loss.item()


# -----------------------------------------------------------------------------
# Simple CLI (checkpoint path currently hard‑coded)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    checkpoint = "/home/paperspace/dev/modded-nanogpt/logs/gptm_record/state_step000125.pt"

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", rank=0, world_size=1)
    best_alpha, best_loss = armijo_search(
        checkpoint_path=checkpoint,
        rank=rank,
        device="cuda",
        world_size=8,
    )

    print(f"Chosen alpha={best_alpha:.6f}, validation loss={best_loss:.6f}")

    dist.destroy_process_group()
