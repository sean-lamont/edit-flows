import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import cast
from torchtyping import TensorType as T
from pathlib import Path
from matplotlib.animation import FuncAnimation

from constants import PAD_TOKEN, MASK_TOKEN
from data import make_batch
from utils import apply_ins_del_operations, get_adaptive_h, pretty_print, opt_align_xs_to_zs
from flows import CubicScheduler, EmptyCoupling

def poisson_apply_ins_del_operations(
        x_t: torch.Tensor,
        ins_vals: torch.Tensor,
        del_mask: torch.Tensor,
        max_seq_len: int = 512,
        pad_token=PAD_TOKEN,
) -> torch.Tensor:
    """
    Apply insertion and deletion operations to a sequence x_t based on the provided masks.
    """
    batch_size, seq_len = x_t.shape
    device = x_t.device

    # Handle simultaneous ins+del as substituting a mask
    replace_mask = (ins_vals > 0) & del_mask
    x_t_modified = x_t.clone()
    x_t_modified[replace_mask] = MASK_TOKEN
    # subtract 1 from inserts
    ins_vals[replace_mask] -= 1


    # Update ins/del masks after handling replacements
    # eff_ins_mask = ins_mask & ~replace_mask
    del_mask = del_mask & ~replace_mask

    # Compute new lengths after applying ins/del operations
    xt_pad_mask = (x_t == pad_token)  # (batch_size, seq_len)
    xt_seq_lens = (~xt_pad_mask).sum(dim=1)  # (batch_size,)
    
    new_lengths = xt_seq_lens + ins_vals.sum(dim=1) - del_mask.sum(dim=1)
    
    max_new_len = int(new_lengths.max().item())

    if max_new_len <= 0:
        print(f"Unexpected max_new_len <= 0: {max_new_len}, did we delete everything?")
        return torch.full((batch_size, 1), pad_token, dtype=x_t.dtype, device=device)

    # Pre-allocate result
    x_new = torch.full((batch_size, max_new_len), pad_token, dtype=x_t.dtype, device=device)

    # Compute positions
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)  # (batch_size, 1)
    pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
    
    cum_del = torch.cumsum(del_mask, dim=1)  # num del up to & incl. current pos
    
    cum_ins = torch.cumsum(ins_vals, dim=1)  # num ins up to & incl. current pos
    
    cum_ins_before = F.pad(cum_ins[:, :-1], (1, 0), value=0)  # num ins before current pos

    # Place non-deleted tokens
    new_pos = pos_idx + cum_ins_before - cum_del # new pos of tokens shifted by ins/del
    
    keep_mask = ~del_mask & (new_pos >= 0) & (new_pos < max_new_len)  # tokens to keep (non-deleted)
    
    if keep_mask.any():
        x_new[batch_idx.expand(-1, seq_len)[keep_mask], new_pos[keep_mask].long()] = x_t_modified[keep_mask]

    # Place insertions
    if (ins_vals > 0).any():
        # Vectorized approach to handle multiple insertions per position
        # 1. Find all locations that need insertions
        ins_b, ins_p = (ins_vals > 0).nonzero(as_tuple=True)
        
        # 2. Get the number of insertions and base positions for these locations
        num_insertions_at_loc = ins_vals[ins_b, ins_p].long()
        base_positions = new_pos[ins_b, ins_p]

        # 3. Repeat batch indices and base positions for each insertion
        total_insertions = num_insertions_at_loc.sum()
        if total_insertions > 0:
            repeated_batch_indices = ins_b.repeat_interleave(num_insertions_at_loc)
            repeated_base_pos = base_positions.repeat_interleave(num_insertions_at_loc)

            # 4. Generate insertion offsets (1, 2, ..., n) for each location
            # This creates a sequence [1, 1, 2, 1, 2, 3, ...] for ins_vals [1, 2, 3, ...]
            is_start_of_sequence = torch.cat([torch.tensor([True], device=device), num_insertions_at_loc[:-1] > 0])
            offsets = (~is_start_of_sequence.repeat_interleave(num_insertions_at_loc)).cumsum(dim=0) + 1

            # 5. Calculate final insertion positions and create a mask for valid positions
            final_ins_pos = repeated_base_pos + offsets.to(repeated_base_pos.device)
            valid_mask = (final_ins_pos >= 0) & (final_ins_pos < max_new_len)
            x_new[repeated_batch_indices[valid_mask], final_ins_pos[valid_mask]] = MASK_TOKEN

    if max_new_len > max_seq_len:
        print(f"Warning: max_new_len {max_new_len} exceeds max_seq_len {max_seq_len}, truncating.")
        max_new_len = max_seq_len

    return x_new[:, :max_new_len]

def run_sampling(model, device: torch.device, V: int, step: int = 0):
    n_steps = 100
    n_samples = 4
    t_min = 0.01

    model.eval()

    default_h = 1 / n_steps
    t = t_min * torch.ones(n_samples, 1, device=device)

    # Sample initial x_t = x_0 from the coupling
    min_seq_len = 128
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

    x_t = x_0.clone().to(device)
    x_pad_mask = (x_t == PAD_TOKEN)  # Create padding mask for x_t
    x_ts = [x_t.clone()]

    # just use default sampler for now (todo maybe change later)
    scheduler = CubicScheduler(a=1.0, b=1.0) # Re-initialize scheduler for sampling

    with tqdm(desc="Euler Sampling") as pbar:
        while t.max() <= 1 - default_h:
            u_t, sub_probs = model.forward(
                cast(T["batch_size", "x_seq_len", "long"], x_t.to(device)),
                cast(T["batch_size", 1, "float"], t.to(device)),
                cast(T["batch_size", "x_seq_len", "bool"], x_pad_mask.to(device)),
            )
            lambda_ins = u_t[:, :, 0]  # Insertion rate        (n_samples, x_seq_len)
            lambda_sub = u_t[:, :, 1]  # Substitution rate     (n_samples, x_seq_len)
            lambda_del = u_t[:, :, 2]  # Deletion rate         (n_samples, x_seq_len)

            adapt_h = get_adaptive_h(default_h, t, scheduler)

            # Sample insertions and deletion/substitutions based on rates
            # ins_mask = torch.rand(
            #     size=lambda_ins.shape, device=lambda_ins.device) < 1 - torch.exp(-adapt_h * lambda_ins)

            ins_vals = torch.poisson(adapt_h * lambda_ins).long()


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
            sub_tokens = torch.full(sub_probs.shape[:2], PAD_TOKEN, dtype=torch.long, device=device)

            non_pad_mask = ~x_pad_mask

            if non_pad_mask.any():
                sub_sampled = torch.multinomial(sub_probs[non_pad_mask], num_samples=1, replacement=True).squeeze(-1)
                sub_tokens[non_pad_mask] = sub_sampled

            # Apply operations based on masks
            x_t[sub_mask] = sub_tokens[sub_mask]

            x_t = poisson_apply_ins_del_operations(
                cast(T["batch_size", "seq_len", "long"], x_t),
                cast(T["batch_size", "seq_len", "long"], ins_vals),
                cast(T["batch_size", "seq_len", "bool"], del_mask),
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
