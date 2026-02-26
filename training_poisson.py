import hashlib
import sys
from collections import defaultdict
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import lightning.pytorch as pl
from torch.utils.data import IterableDataset, DataLoader
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torchtyping import TensorType as T
from tqdm import tqdm

from config import L
from constants import PAD_TOKEN, GAP_TOKEN, MASK_TOKEN, BOS_TOKEN
from data import sample_cond_pt, make_x0_with_bounds, make_batch, make_ut_mask_from_z, \
    fill_gap_tokens_with_repeats
from flows import CubicScheduler, EmptyCoupling, x2prob, UniformCoupling, GeneratorCoupling, sample_p
from model import SimpleEditFlowsTransformer
from poisson_model import PoissonEditFlowsTransformer
from sampling_poisson import run_sampling
from utils import opt_align_xs_to_zs, pretty_print, safe_chr, rm_gap_tokens


def poisson_make_uz_mask(
        z_t: torch.Tensor,
        z_1: torch.Tensor,
        vocab_size: int = 130,
        pad_token: int = PAD_TOKEN,
        gap_token: int = GAP_TOKEN,
) -> torch.Tensor:
    """
    Create a mask for u_cat for indexing the output rate tensor based on differences between z_t and z_1.
    For each position i where z_t and z_1 differ, we index as follows:

    - z_t[i] = GAP_TOKEN & z_1[i] = c => u_mask[i, insert] = 1
    - z_t[i] = c & z_1[i] = GAP_TOKEN => u_mask[i, delete] = 1
    - z_t[i] = c1 & z_1[i] = c2 => u_mask[i, substitute, c1, c2] = 1
    """
    batch_size, z_seq_len = z_t.shape
    n_ops = vocab_size + 2  # substitute + delete + insert

    z_neq = (z_t != z_1) & (z_t != pad_token) & (z_1 != pad_token)
    z_ins = (z_t == gap_token) & (z_1 != gap_token) & z_neq  # (batch_size, z_seq_len)
    z_del = (z_t != gap_token) & (z_1 == gap_token) & z_neq  # (batch_size, z_seq_len)
    z_sub = z_neq & ~z_ins & ~z_del  # (batch_size, z_seq_len)

    # mask (batch_size, z_seq_len, u_ops) where 1 indicates operation that bring z_t closer to z_1
    u_mask = torch.zeros((batch_size, z_seq_len, n_ops), dtype=torch.bool, device=z_t.device)
    # u_mask[z_ins, z_1[z_ins]] = True
    u_mask[z_sub, z_1[z_sub]] = True
    u_mask[:, :, -1][z_del] = True
    u_mask[:, :, -2][z_ins] = True

    assert z_neq.sum() == (z_ins | z_del | z_sub).sum(), "Mismatch in number of edits"
    assert z_neq.sum() == u_mask.sum(), "Mismatch in number of edits in mask"

    return u_mask


# sample with scheduler prob for each value whether to be z_0, z_1, or [mask].
def sample_zt(z_0, z_1, mask_scheduler, default_scheduler, t, V):
    z_neq = (z_0 != z_1) & (z_0 != PAD_TOKEN) & (z_1 != PAD_TOKEN)
    z_ins = (z_0 == GAP_TOKEN) & (z_1 != GAP_TOKEN) & z_neq  # (batch_size, z_seq_len)

    # t orig = (batch_size, 1) -> (batch_size, 1, 1)
    t = t.reshape(-1, 1, 1)


    mask_t = mask_scheduler(t)
    default_t = default_scheduler(t)

    # one-hot vecs (b, s, v)
    p_0 = x2prob(z_0, V + 4)
    p_1 = x2prob(z_1, V + 4)
    p_mask = x2prob(torch.tensor([MASK_TOKEN]).expand_as(z_0).to(z_0.device), V + 4)

    # for insert
    pt_ins = (1 - mask_t) * p_0 \
             + mask_t * (1 - default_t) * p_mask \
             + mask_t * default_t * p_1

    # for delete/sub
    pt = (1 - default_t) * p_0 + default_t * p_1

    pt = torch.where(z_ins.unsqueeze(-1), pt_ins, pt)

    return sample_p(pt)


class EditFlowsIterableDataset(IterableDataset):
    def __init__(self, V, batch_size, min_seq_len, max_seq_len, coupling, seq_align_fn, num_cycles_fn, x_int_fn):
        super().__init__()
        self.V = V
        self.batch_size = batch_size
        self.min_length = min_seq_len
        self.max_length = max_seq_len
        self.coupling = coupling
        self.seq_align_fn = seq_align_fn
        self.num_cycles_fn = num_cycles_fn
        self.x_int_fn = x_int_fn

    def __iter__(self):
        while True:
            yield make_batch(
                batch_size=self.batch_size,
                min_length=self.min_length,
                max_length=self.max_length,
                vocab_size=self.V,
                coupling=self.coupling,
                seq_align_fn=self.seq_align_fn,
                num_cycles_fn=self.num_cycles_fn,
                x_int_fn=self.x_int_fn,
            )


class SamplingCallback(Callback):
    def __init__(self, V, every_n_steps=500):
        super().__init__()
        self.V = V
        self.every_n_steps = every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.every_n_steps == 0:
            print(f"\nRunning sampling at step {trainer.global_step + 1}")
            pl_module.eval()
            run_sampling(pl_module.model, pl_module.device, self.V, trainer.global_step + 1)
            pl_module.train()


class MetricsPlottingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.metrics = defaultdict(list)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for key, value in trainer.logged_metrics.items():
            if 'epoch' in key:
                continue
            if isinstance(value, torch.Tensor):
                self.metrics[key].append(value.item())
            else:
                self.metrics[key].append(value)

    def on_train_end(self, trainer, pl_module):
        # Plotting logic
        plt.figure(figsize=(18, 5))

        # 1. Plot loss (raw and smoothed)
        plt.subplot(1, 3, 1)
        loss_metrics = self.metrics.get("loss", [])
        if loss_metrics:
            plt.plot(loss_metrics, label='Raw Loss', color='lightblue', alpha=0.6)
            smoothed_losses = pd.Series(loss_metrics).ewm(alpha=0.1).mean()
            plt.plot(smoothed_losses, label='Smoothed Loss (EMA)', color='blue', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 2. Plot u_ins, u_del, u_sub
        plt.subplot(1, 3, 2)
        plt.plot(self.metrics.get("u_ins", []), label='u_ins', color='green')
        plt.plot(self.metrics.get("u_del", []), label='u_del', color='red')
        plt.plot(self.metrics.get("u_sub", []), label='u_sub', color='purple')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title('u_ins, u_del, u_sub Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 3. Plot u_tot and u_con
        plt.subplot(1, 3, 3)
        plt.plot(self.metrics.get("u_tot", []), label='u_tot', color='orange')
        plt.plot(self.metrics.get("u_con", []), label='u_con', color='brown')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.title('u_tot and u_con Over Time')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig("metrics.png", dpi=300, bbox_inches='tight')
        plt.close()


class EditFlowsLightningModule(pl.LightningModule):
    def __init__(self, V: int, learning_rate: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = PoissonEditFlowsTransformer(
            vocab_size=V + 3,  # +3 for PAD + BOS tokens, MASK
            hidden_dim=512,
            num_layers=8,
            num_heads=32,
            max_seq_len=2 * L,
            pad_token_id=PAD_TOKEN,
            bos_token_id=BOS_TOKEN,
        )

        self.V = V + 1
        self.learning_rate = learning_rate

        self.mask_scheduler = CubicScheduler(a=3.0, b=0.0)
        self.default_scheduler = CubicScheduler(a=1.0, b=1.0)

    def training_step(self, batch, batch_idx):
        x_0, x_1, z_0, z_1, t, _ = batch

        z_t = sample_zt(z_0, z_1, self.mask_scheduler, self.default_scheduler, t, self.V)
        x_t, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z_t)
        assert (~x_pad_mask).sum(1).max().item() == x_t.shape[1]

        uz_mask = poisson_make_uz_mask(
            cast(T["batch_size", "z_seq_len", "long"], z_t),
            cast(T["batch_size", "z_seq_len", "long"], z_1),
            vocab_size=self.V + 2,
        )

        u_t, sub_probs = self.model.forward(
            tokens=cast(T["batch_size", "x_seq_len", "long"], x_t),
            time_step=cast(T["batch_size", 1, "float"], t),
            padding_mask=cast(T["batch_size", "x_seq_len", "bool"], x_pad_mask),
        )
        lambda_ins = u_t[:, :, 0]
        lambda_sub = u_t[:, :, 1]
        lambda_del = u_t[:, :, 2]

        u_tia_sub = lambda_sub.unsqueeze(-1) * sub_probs
        u_tia_ins = lambda_ins.unsqueeze(-1)
        u_tia_del = lambda_del.unsqueeze(-1)

        ux_cat = torch.cat([u_tia_sub, u_tia_ins, u_tia_del], dim=-1)
        uz_cat = fill_gap_tokens_with_repeats(ux_cat, z_gap_mask, z_pad_mask)
        u_tot = u_t.sum(dim=(1, 2))

        if torch.isnan(ux_cat).any():
            raise ValueError("NaN detected in ux_cat")
        if torch.isnan(uz_cat).any():
            raise ValueError("NaN detected in uz_cat")

        default_coeff = (self.default_scheduler.derivative(t) / (1 - self.default_scheduler(t))).to(self.device)
        ins_coeff = (self.mask_scheduler.derivative(t) / (1 - self.mask_scheduler(t))).to(self.device)

        log_uz_cat = torch.clamp(uz_cat.log(), min=-20)

        z_neq = (z_0 != z_1) & (z_0 != PAD_TOKEN) & (z_1 != PAD_TOKEN)
        z_ins = (z_0 == GAP_TOKEN) & (z_1 != GAP_TOKEN) & z_neq

        sched_coeff = torch.where(z_ins.to(self.device), ins_coeff, default_coeff)

        loss = u_tot - (log_uz_cat * uz_mask.to(self.device) * sched_coeff.unsqueeze(-1)).sum(dim=(1, 2))
        loss = loss.mean()

        assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN or Inf"

        u_ins = lambda_ins.sum(dim=1).mean()
        u_del = lambda_del.sum(dim=1).mean()
        u_sub = lambda_sub.sum(dim=1).mean()
        u_con = (uz_cat * uz_mask.to(self.device)).sum(dim=(1, 2)).mean()

        self.log_dict({
            "loss": loss,
            "u_tot": u_tot.mean(),
            "u_ins": u_ins,
            "u_del": u_del,
            "u_sub": u_sub,
            "u_con": u_con,
        }, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyperparameters
    V = 128
    batch_size = 128
    min_seq_len = 128
    max_seq_len = 128
    steps = 4000000 // batch_size
    
    # Data components
    seq_align_fn = opt_align_xs_to_zs
    num_cycles_fn = lambda: 3.5
    x_int_fn = lambda: 0
    coupling = EmptyCoupling()

    # Dataset and DataLoader
    dataset = EditFlowsIterableDataset(
        V=V,
        batch_size=batch_size,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
        coupling=coupling,
        seq_align_fn=seq_align_fn,
        num_cycles_fn=num_cycles_fn,
        x_int_fn=x_int_fn,
    )
    # batch_size=None because the dataset yields entire batches
    dataloader = DataLoader(dataset, batch_size=None)

    # Lightning Module
    model = EditFlowsLightningModule(V=V)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='loss',
        dirpath='.',
        filename='best_model',
        save_top_k=1,
        mode='min',
        every_n_train_steps=100,
        save_weights_only=True,
    )
    sampling_callback = SamplingCallback(V=V, every_n_steps=500)
    plotting_callback = MetricsPlottingCallback()

    # Trainer
    trainer = pl.Trainer(
        max_steps=steps,
        callbacks=[checkpoint_callback, sampling_callback, plotting_callback],
        accelerator="auto",
        devices="auto",
    )

    # Start training
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    main()
