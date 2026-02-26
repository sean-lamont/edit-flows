import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchtyping import TensorType as T
from tqdm import tqdm

from constants import PAD_TOKEN
from data import sample_cond_pt, make_x0_with_bounds, make_batch, make_ut_mask_from_z, \
    fill_gap_tokens_with_repeats
from flows import CubicScheduler, EmptyCoupling, x2prob, UniformCoupling, GeneratorCoupling
from model import SimpleEditFlowsTransformer
from sampling import run_sampling
from utils import opt_align_xs_to_zs, pretty_print, safe_chr, rm_gap_tokens, shifted_align_xs_to_zs



def train_model(model: SimpleEditFlowsTransformer, optim: torch.optim.Adam, device: torch.device, V: int,
                save_dir=Path('')):
    # torch.manual_seed(42)
    # np.random.seed(42)

    metrics = defaultdict(list)

    batch_size = 128
    min_seq_len = 128
    max_seq_len = 128


    # seq_align_fn = opt_align_xs_to_zs
    seq_align_fn = shifted_align_xs_to_zs

    # num_cycles_fn = lambda: np.random.uniform(2.5, 4)

    num_cycles_fn = lambda: np.random.uniform(2.5, 4)
    x_int_fn = lambda: np.random.uniform(0, 2 * np.pi)

    # num_cycles_fn = lambda: 3.5
    # x_int_fn = lambda: 0

    generator_fn = lambda x1: make_x0_with_bounds(batch_size=int(x1.shape[0]), min_length=min_seq_len,
                                                  max_length=max_seq_len,
                                                  vocab_size=V, pad_token=PAD_TOKEN,
                                                  num_cycles_fn=lambda: np.random.uniform(1., 2.5), x_int_fn=x_int_fn)

    # generator_fn = lambda x1: make_x0_like_x1(
    #     x1, vocab_size=V, pad_token=PAD_TOKEN, num_cycles_fn=lambda: np.random.uniform(1, 2.5), x_int_fn=x_int_fn)

    # coupling = EmptyCoupling()

    coupling = GeneratorCoupling(generator_fn=generator_fn)

    # coupling = ExtendedCoupling(n_insert=64, vocab_size=V, pad_token=PAD_TOKEN)

    # coupling = UniformCoupling(
    #     min_len=min_seq_len, max_len=max_seq_len, mirror_len=True, vocab_size=V, pad_token=PAD_TOKEN)

    scheduler = CubicScheduler(a=1.0, b=1.0)
    
    model.to(device)
    model.train()

    steps = 4000000 // batch_size
    # steps = 400 // batch_size
    pbar = tqdm(range(steps), desc="Training Edit Flows", unit="step")
    best_avg_loss = float('inf')

    for step in pbar:
        # samples batch of pairs, timestep, and aligns to Z space
        x_0, x_1, z_0, z_1, t, _ = make_batch(
            batch_size=batch_size,
            min_length=min_seq_len,
            max_length=max_seq_len,
            vocab_size=V,
            coupling=coupling,
            seq_align_fn=seq_align_fn,
            num_cycles_fn=num_cycles_fn,
            x_int_fn=x_int_fn,
        )

        z_t = sample_cond_pt(x2prob(z_0, V + 3), x2prob(z_1, V + 3), t, scheduler)  # interpolates in Z space
        x_t, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z_t)  # removes gap tokens and pads to max length
        assert (~x_pad_mask).sum(1).max().item() == x_t.shape[1]

        # uz_mask indicates which operations bring z_t closer to z_1 (batch_size, z_seq_len, 2 * vocab_size + 1)
        uz_mask = make_ut_mask_from_z(
            cast(T["batch_size", "z_seq_len", "long"], z_t),
            cast(T["batch_size", "z_seq_len", "long"], z_1),
            vocab_size=V + 2,  # +2 for PAD and BOS tokens
        )

        # Feeds x_t, t to the model to obtain rates and probabilities for edit operations
        u_t, ins_probs, sub_probs = model.forward(
            tokens=cast(T["batch_size", "x_seq_len", "long"], x_t.to(device)),
            time_step=cast(T["batch_size", 1, "float"], t.to(device)),
            padding_mask=cast(T["batch_size", "x_seq_len", "bool"], x_pad_mask.to(device)),
        )
        lambda_ins = u_t[:, :, 0]  # Insertion rate        (batch_size, x_seq_len)
        lambda_sub = u_t[:, :, 1]  # Substitution rate     (batch_size, x_seq_len)
        lambda_del = u_t[:, :, 2]  # Deletion rate         (batch_size, x_seq_len)

        u_tia_ins = lambda_ins.unsqueeze(-1) * ins_probs  # (batch_size, x_seq_len, vocab_size)
        u_tia_sub = lambda_sub.unsqueeze(-1) * sub_probs  # (batch_size, x_seq_len, vocab_size)
        u_tia_del = lambda_del.unsqueeze(-1)  # (batch_size, x_seq_len, 1)

        ux_cat = torch.cat([u_tia_ins, u_tia_sub, u_tia_del], dim=-1)  # (batch_size, x_seq_len, 2 * vocab_size + 1)
        uz_cat = fill_gap_tokens_with_repeats(ux_cat, z_gap_mask, z_pad_mask)  # (batch_size, z_seq_len, 2 * vocab_size + 1)
        u_tot = u_t.sum(dim=(1, 2))  # (batch_size,)

        if torch.isnan(ux_cat).any():
            raise ValueError("NaN detected in ux_cat")
        if torch.isnan(uz_cat).any():
            raise ValueError("NaN detected in uz_cat")

        # Compute Bregman divergence loss
        sched_coeff = (scheduler.derivative(t) / (1 - scheduler(t))).to(device)
        log_uz_cat = torch.clamp(uz_cat.log(), min=-20)
        loss = u_tot - (log_uz_cat * uz_mask.to(device) * sched_coeff.unsqueeze(-1)).sum(dim=(1, 2))
        loss = loss.mean()

        assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN or Inf"

        optim.zero_grad()
        loss.backward()
        optim.step()

        u_ins = lambda_ins.sum(dim=1).mean().detach().cpu()
        u_del = lambda_del.sum(dim=1).mean().detach().cpu()
        u_sub = lambda_sub.sum(dim=1).mean().detach().cpu()
        u_con = (uz_cat * uz_mask.to(device)).sum(dim=(1, 2)).mean().detach().cpu()

        metrics["loss"].append(loss.item())
        metrics["u_tot"].append(u_tot.mean().item())
        metrics["u_ins"].append(u_ins.item())
        metrics["u_del"].append(u_del.item())
        metrics["u_sub"].append(u_sub.item())
        metrics["u_con"].append(u_con.item())

        if step > 0 and step % 100 == 0:
            avg_loss = np.mean(metrics["loss"][-100:])
            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                torch.save(model.state_dict(), save_dir / Path('best_model.pt'))
                print(f"\nNew best model saved with avg loss {avg_loss:.4f}")
            
            pbar.set_description(f"Avg Loss: {avg_loss:.4f}")
            print(f"Step {step}: Avg Loss (100 steps) = {avg_loss:.4f}, "
                  f"u_tot = {u_tot.mean().item():.4f}, "
                  f"u_ins = {u_ins:.4f}, "
                  f"u_del = {u_del:.4f}, "
                  f"u_sub = {u_sub:.4f}, "
                  f"u_con = {u_con:.4f}")

        if step > 0 and step % 500 == 0:
            run_sampling(model, device, V, step, save_dir=save_dir)
            model.train()

    # Plotting metrics
    plt.figure(figsize=(18, 5))

    # 1. Plot loss (raw and smoothed)
    plt.subplot(1, 3, 1)
    plt.plot(metrics["loss"], label='Raw Loss', color='lightblue', alpha=0.6)
    smoothed_losses = pd.Series(metrics["loss"]).ewm(alpha=0.1).mean()
    plt.plot(smoothed_losses, label='Smoothed Loss (EMA)', color='blue', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 2. Plot u_ins, u_del, u_sub
    plt.subplot(1, 3, 2)
    plt.plot(metrics["u_ins"], label='u_ins', color='green')
    plt.plot(metrics["u_del"], label='u_del', color='red')
    plt.plot(metrics["u_sub"], label='u_sub', color='purple')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('u_ins, u_del, u_sub Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 3. Plot u_tot and u_con
    plt.subplot(1, 3, 3)
    plt.plot(metrics["u_tot"], label='u_tot', color='orange')
    plt.plot(metrics["u_con"], label='u_con', color='brown')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.title('u_tot and u_con Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig("metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

    return model, optim