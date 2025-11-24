import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import cast
from torchtyping import TensorType as T
from pathlib import Path
from matplotlib.animation import FuncAnimation

from constants import PAD_TOKEN
from data import make_batch
from model import SimpleEditFlowsTransformer
from utils import apply_ins_del_operations, get_adaptive_h, pretty_print, opt_align_xs_to_zs
from flows import CubicScheduler, EmptyCoupling


def run_sampling(model: SimpleEditFlowsTransformer, device: torch.device, V: int, step: int = 0):
    n_steps = 1000
    n_samples = 4
    t_min = 0.0

    model.eval()

    default_h = 1 / n_steps
    t = t_min * torch.ones(n_samples, 1)

    # Sample initial x_t = x_0 from the coupling
    min_seq_len = 64
    max_seq_len = 128
    num_cycles_fn = lambda: np.random.uniform(2.5, 4)
    x_int_fn = lambda: np.random.uniform(0, 2 * np.pi)
    coupling = EmptyCoupling()
    seq_align_fn = opt_align_xs_to_zs

    x_0, _, _, _, _, _ = make_batch(
        batch_size=n_samples,
        min_length=min_seq_len,
        max_length=max_seq_len,
        vocab_size=V,
        coupling=coupling,
        seq_align_fn=seq_align_fn,
        num_cycles_fn=num_cycles_fn,
        x_int_fn=x_int_fn,
    )

    x_t = x_0.clone()
    x_pad_mask = (x_t == PAD_TOKEN)  # Create padding mask for x_t
    x_ts = [x_t.clone()]

    scheduler = CubicScheduler(a=1.0, b=1.0) # Re-initialize scheduler for sampling

    with tqdm(desc="Euler Sampling") as pbar:
        while t.max() <= 1 - default_h:
            u_t, ins_probs, sub_probs = model.forward(
                cast(T["batch_size", "x_seq_len", "long"], x_t.to(device)),
                cast(T["batch_size", 1, "float"], t.to(device)),
                cast(T["batch_size", "x_seq_len", "bool"], x_pad_mask.to(device)),
            )
            lambda_ins = u_t[:, :, 0].cpu()  # Insertion rate        (n_samples, x_seq_len)
            lambda_sub = u_t[:, :, 1].cpu()  # Substitution rate     (n_samples, x_seq_len)
            lambda_del = u_t[:, :, 2].cpu()  # Deletion rate         (n_samples, x_seq_len)

            adapt_h = get_adaptive_h(default_h, t, scheduler)

            # Sample insertions and deletion/substitutions based on rates
            ins_mask = torch.rand(
                size=lambda_ins.shape, device=lambda_ins.device) < 1 - torch.exp(-adapt_h * 0.04 * lambda_ins)
            del_sub_mask = torch.rand(
                size=lambda_sub.shape, device=lambda_sub.device
            ) < 1 - torch.exp(-adapt_h * (lambda_sub + lambda_del))

            # For deletion/substitution, sample based on the relative rates
            prob_del = torch.where(
                del_sub_mask, lambda_del / (lambda_sub + lambda_del), torch.zeros_like(lambda_del))
            del_mask = torch.bernoulli(prob_del).bool()
            sub_mask = del_sub_mask & ~del_mask
            assert sub_mask.sum() + del_mask.sum() == del_sub_mask.sum()

            # Only sample tokens for non-pad positions, fill pad positions with PAD_TOKEN
            ins_tokens = torch.full(ins_probs.shape[:2], PAD_TOKEN, dtype=torch.long)
            sub_tokens = torch.full(sub_probs.shape[:2], PAD_TOKEN, dtype=torch.long)
            non_pad_mask = ~x_pad_mask
            if non_pad_mask.any():
                ins_sampled = torch.multinomial(ins_probs[non_pad_mask].cpu(), num_samples=1, replacement=True).squeeze(-1)
                sub_sampled = torch.multinomial(sub_probs[non_pad_mask].cpu(), num_samples=1, replacement=True).squeeze(-1)
                ins_tokens[non_pad_mask] = ins_sampled
                sub_tokens[non_pad_mask] = sub_sampled

            # Apply operations based on masks
            x_t[sub_mask] = sub_tokens[sub_mask]
            x_t = apply_ins_del_operations(
                cast(T["batch_size", "seq_len", "long"], x_t),
                cast(T["batch_size", "seq_len", "bool"], ins_mask),
                cast(T["batch_size", "seq_len", "bool"], del_mask),
                cast(T["batch_size", "seq_len", "long"], ins_tokens),
                max_seq_len=model.max_seq_len,
                pad_token=PAD_TOKEN,
            )
            x_pad_mask = (x_t == PAD_TOKEN)  # Update padding mask after operations

            t = t + adapt_h
            x_ts.append(x_t.clone())
            pbar.update(1)

    # Visualize the sampled sequences
    n_seqs_to_plot = 5
    seq_indices = np.linspace(0, len(x_ts) - 1, n_seqs_to_plot, dtype=int)

    fig, ax = plt.subplots(n_samples, n_seqs_to_plot, figsize=(20, 4 * n_samples), sharey=True)

    for j in range(n_samples):
        pretty_print(x_ts[0][j], compact=True)
        pretty_print(x_ts[-1][j], compact=True)
        print()

        seqs = [x_ts[i][j].cpu().numpy().squeeze() for i in seq_indices]

        for i, seq in enumerate(seqs):
            y = seq[seq != PAD_TOKEN]  # Remove padding token
            x = np.arange(len(y))
            ax[j, i].scatter(x, y, label=f"Step {seq_indices[i]}", s=10)
            ax[j, i].set_title(f"Step {seq_indices[i]}")
            ax[j, i].set_xlabel("Index")
            ax[j, i].set_ylabel("Token Value")
            ax[j, i].grid(True, alpha=0.3)
            ax[j, i].legend()

    plt.tight_layout()
    plt.savefig(f"model_samples_step_{step}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # # Make video animation of the edit flows through sampling steps
    # fig, axes = plt.subplots(1, n_samples, figsize=(6 * n_samples, 6), sharey=True)
    # if n_samples == 1:
    #     axes = [axes]

    # def update_all(frame):
    #     for sample_idx in range(n_samples):
    #         ax = axes[sample_idx]
    #         ax.clear()
    #         y = x_ts[frame][sample_idx].cpu().numpy().squeeze()
    #         y = y[y != PAD_TOKEN]  # Remove padding token
    #         x = np.arange(len(y))
    #         ax.scatter(x, y, s=10)
    #         ax.set_title(f"Sample {sample_idx} - Step {frame}")
    #         ax.set_xlabel("Index")
    #         ax.set_ylabel("Token Value")
    #         ax.grid(True, alpha=0.3)
    #         ax.legend([f"Time Step: {frame}"])

    # anim = FuncAnimation(fig, update_all, frames=len(x_ts), repeat=False, interval=100)
    # anim.save("editflow_anim.mp4", writer='ffmpeg', fps=n_steps // 10)
    # plt.close(fig)
