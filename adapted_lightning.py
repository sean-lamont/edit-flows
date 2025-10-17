import lightning.pytorch as pl
from scheduler import CubicScheduler
from utils import *
from torch.nn import functional as F
import time
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

class AdaptedLitModule(pl.LightningModule):
    def __init__(self, model: nn.Module, full_vocab_size, pad_token_id, gap_token_id, lr=1e-4, scheduler_cfg=None, anneal_end_step=10000):
        super().__init__()
        # self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.kappa = CubicScheduler(**(scheduler_cfg or {'a': 1.0, 'b': 1.0}))
        self.anneal_end_step = anneal_end_step
        self.full_vocab_size = full_vocab_size
        self.pad_token = pad_token_id
        self.gap_token = gap_token_id
        self.lr = lr

    def configure_optimizers(self):
        return DeepSpeedCPUAdam(self.parameters(), lr=self.lr, eps=1e-6)
        # return DeepSpeedCPUAdam(self.parameters(), 1e-5, eps=1e-6)
        # return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

    def forward(self, tokens, t, pad_mask, attn_mask_ratio):
        return self.model(tokens, t, pad_mask, attn_mask_ratio)

    def _loss(self, batch):
        x1, x0, z0, z1, t, context_lens = batch['x1'], batch['x0'], batch['z0'], batch['z1'], batch['t'], batch['context_lens']

        p0 = x2prob(z0, self.full_vocab_size)
        p1 = x2prob(z1, self.full_vocab_size)

        zt = sample_cond_pt(p0, p1, t, self.kappa)

        xt, x_pad, z_gap, z_pad = rm_gap_tokens(zt, self.pad_token, self.gap_token)

        attn_mask_ratio = min(1.0, self.global_step / self.anneal_end_step)

        rates, ins_probs, sub_probs = self(xt, t, x_pad, attn_mask_ratio)

        # todo adjust to ignore context, update max length in data loader

        lam_ins = rates[:, :, 0]
        lam_sub = rates[:, :, 1]
        lam_del = rates[:, :, 2]

        ux_ins = lam_ins.unsqueeze(-1) * ins_probs
        ux_sub = lam_sub.unsqueeze(-1) * sub_probs
        ux_del = lam_del.unsqueeze(-1)

        ux_cat = torch.cat([ux_ins, ux_sub, ux_del], dim=-1)

        uz_cat = fill_gap_tokens_with_repeats(ux_cat, z_gap, z_pad)

        uz_mask = make_ut_mask_from_z(zt, z1, self.full_vocab_size, self.pad_token, self.gap_token)

        u_tot = rates.sum(dim=(1, 2))

        sched_coeff = (self.kappa.derivative(t) / (1 - self.kappa(t))).to(self.device)

        log_uz_cat = torch.clamp(uz_cat.log(), min=-20)

        loss_vec = u_tot - (log_uz_cat * uz_mask * sched_coeff.unsqueeze(-1)).sum(dim=(1, 2))

        return loss_vec.mean(), {
            'u_tot': u_tot.mean(),
            'u_ins': lam_ins.sum(1).mean(),
            'u_sub': lam_sub.sum(1).mean(),
            'u_del': lam_del.sum(1).mean(),
            'attn_mask_ratio': attn_mask_ratio,
        }

    def training_step(self, batch, batch_idx):
        loss, metrics = self._loss(batch)
        self.log('train/loss', loss, prog_bar=True)
        for k, v in metrics.items():
            self.log(f'train/{k}', v, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self._loss(batch)
        self.log('val/loss', loss, prog_bar=True)
        for k, v in metrics.items():
            self.log(f'val/{k}', v, prog_bar=False)
