"""
train_gptm_armijo.py  —  concise edition
========================================
One‑dimensional line‑search that finds the **validation‑loss‑minimising** step
size α along the combined optimiser update direction **D**.

* supports multi‑GPU via NCCL (each rank → its own GPU)
* keeps a **single** model copy on device; weights + D live on CPU
* works with *any* Muon checkpoint (adds/casts missing `mantissa` buffers)
* minimisation = bracket + Golden‑section search (tol = 1e‑4)
"""

from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, gc, torch, torch.distributed as dist, torch.nn as nn
from typing import List, Tuple
from gpt_static import (
    Hyperparameters as HP, GPT, Muon, run_validation, load_state_dict_safely,
    distributed_data_generator as ddg, get_window_size_blocks as wsize,
)

# ── distributed bootstrap ───────────────────────────────────────────────────
if not dist.is_initialized():
    dist.init_process_group("nccl", init_method="env://")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(local)
DEV = torch.device(f"cuda:{local}")

hp = HP()

# ── helpers ─────────────────────────────────────────────────────────────────

def build_model() -> GPT:
    m = GPT(hp.vocab_size, 16, 8, 1024, max(hp.train_seq_len, hp.val_seq_len)).to(DEV)
    for p in m.parameters(): p.data = p.data.to(torch.bfloat16)
    return torch.compile(m, dynamic=False)


def build_opts(m: GPT, states: List[dict], step: int) -> List[torch.optim.Optimizer]:
    mats = sorted([p for p in m.blocks.parameters() if p.ndim >= 2], key=lambda x: x.size(), reverse=True)
    o1 = torch.optim.AdamW([
        dict(params=[m.lm_head_w], lr=1/320),
        dict(params=[*m.embed.parameters(), *m.value_embeds.parameters()], lr=0.3),
        dict(params=[m.scalars], lr=0.015),
    ], betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0, fused=True)
    o2 = Muon(mats, lr=0.025, momentum=0.95, rank=rank, world_size=world)
    if states:
        for o, s in zip((o1, o2), states):
            o.load_state_dict(s)
            if isinstance(o, Muon):
                for p, st in o.state.items():
                    st["mantissa"] = st.get("mantissa", torch.zeros_like(p, dtype=torch.uint16)).to(torch.uint16)
            for g in o.param_groups: g["step"] = step
    return [o1, o2]


def compute_direction(m: GPT, opt: List[torch.optim.Optimizer], step: int) -> List[torch.Tensor]:
    w0 = [p.detach().cpu().clone() for p in m.parameters()]
    batch = ddg(hp.train_files, world * hp.train_seq_len, rank, world, seed=step)
    inp, tgt = next(batch)
    [o.zero_grad(set_to_none=True) for o in opt]
    (m(inp.to(DEV), tgt.to(DEV), wsize(step))).backward(); [o.step() for o in opt]
    D = [(p.detach().cpu() - w).to(torch.bfloat16) for p, w in zip(m.parameters(), w0)]
    for p, w in zip(m.parameters(), w0): p.data.copy_(w.to(DEV))  # restore
    if world > 1:  # broadcast via GPU then back to CPU
        buf = [d.to(DEV) if rank == 0 else torch.empty_like(p, device=DEV) for d, p in zip(D, m.parameters())]
        [dist.broadcast(t, 0) for t in buf]
        D = [t.cpu() for t in buf]
    gc.collect(); torch.cuda.empty_cache(); return D


def apply(m: GPT, D: List[torch.Tensor], a: float):
    for p, d in zip(m.parameters(), D): p.data.add_(a, d.to(DEV, dtype=p.dtype))


def val_loss(m: GPT, a: float) -> torch.Tensor:
    apply(m, m._D, a)
    v = run_validation(m, m._step, hp, rank, world, seed=1234 + int(a * 1e4)).detach()
    apply(m, m._D, -a)
    return v

# ── 1‑D minimisation (Golden‑section) ───────────────────────────────────────

from typing import Dict

def minimise(m: GPT) -> Dict[float, float]:
    """Golden‑section search that logs every α evaluated instead of just the best one."""
    phi = (5 ** 0.5 - 1) / 2             # 1/φ
    losses: Dict[float, float] = {}      # α → val_loss(α)

    def record(alpha: float) -> float:
        """Cache and return the validation loss at α."""
        return losses.setdefault(alpha, val_loss(m, alpha))

    # --- bracket the minimiser ------------------------------------------------
    a, b = 0.0, 4.0
    fa, fb = record(a), record(b)
    while fb < fa and b < 128:
        a, b, fa, fb = b, 2 * b, fb, record(2 * b)

    # --- golden‑section shrink ------------------------------------------------
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = record(c), record(d)

    while b - a > 1e-3:
        if fc < fd:        # minimum is in [a,d]
            b, d, fb, fd = d, c, fd, fc
            c = b - phi * (b - a)
            fc = record(c)
        else:              # minimum is in [c,b]
            a, c, fa, fc = c, d, fc, fd
            d = a + phi * (b - a)
            fd = record(d)

    return losses  # mapping of every α we tried to its validation loss


# ── driver ──────────────────────────────────────────────────────────────────

def line_search(ckpt: str) -> Tuple[float, float]:
    state = torch.load(ckpt, map_location="cpu") if rank == 0 else {}
    step = state.get("step", 0) if rank == 0 else 0
    st = torch.tensor([step], device=DEV); dist.broadcast(st, 0); step = int(st)
    m = build_model(); m._step = step
    if rank == 0: load_state_dict_safely(m, state["model"], False)
    [dist.broadcast(p.data, 0) for p in m.parameters()]
    optim = build_opts(m, state.get("optimizers", []) if rank == 0 else [], step)
    m._D = compute_direction(m, optim, step); m.eval()
    return minimise(m)

if __name__ == "__main__":
    # dfs = []
    # for step in range(0, 4001, 1000):
    #     alphas_to_losses = line_search(f"/home/paperspace/dev/modded-nanogpt/logs/gptm_record/state_step{step:06d}.pt")
    #     if rank == 0:
    #         for alpha, loss in alphas_to_losses.items():
    #             print(f"Alpha: {alpha}, Loss: {loss}")
    #     df = pd.DataFrame(alphas_to_losses.items(), columns=["alpha", "loss"])
    #     df["step"] = step
    #     df['loss'] = df['loss'].apply(lambda x: float(x))
    #     dfs.append(df)
    # df = pd.concat(dfs)
    # df.to_csv("armijo.csv", index=False)
    dist.destroy_process_group()

    df = pd.read_csv("armijo.csv")
    # for each step, find the minimizing alpha (called alpha^*) and corresponding loss
    # then plot alpha^* vs. val_loss
    grouped = df.groupby("step")
    minimized_df = grouped.apply(lambda x: x[x["loss"] == x["loss"].min()])
    plt.scatter(minimized_df["loss"], minimized_df["alpha"])
    plt.savefig("armijo_scatter.png")

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a function to update the plot for each frame
    def update(frame):
        ax.clear()
        ax.scatter(df[df["step"] == frame]["alpha"], df[df["step"] == frame]["loss"])
        ax.set_title(f"Step {frame}")
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=df["step"].unique(), repeat=False)
    
    # Save the animation as a GIF
    ani.save("armijo.gif", writer="pillow")
    
    
