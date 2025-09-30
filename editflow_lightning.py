import lightning.pytorch as pl

from scheduler import CubicScheduler
from utils import *


class EditFlowLitModule(pl.LightningModule):
    def __init__(self, model: nn.Module, lr=1e-4, scheduler_cfg=None):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.kappa = CubicScheduler(**(scheduler_cfg or {'a': 1.0, 'b': 1.0}))

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)  # , eps=1e-3)

    def forward(self, tokens, t, pad_mask):
        return self.model(tokens, t, pad_mask)

    def _loss(self, batch):
        x0, x1, z0, z1, t = batch['x0'], batch['x1'], batch['z0'], batch['z1'], batch['t']

        p0 = x2prob(z0, FULL_VOCAB)
        p1 = x2prob(z1, FULL_VOCAB)

        zt = sample_cond_pt(p0, p1, t, self.kappa)

        xt, x_pad, z_gap, z_pad = rm_gap_tokens(zt)

        rates, ins_probs, sub_probs = self(xt, t, x_pad)

        lam_ins = rates[:, :, 0]
        lam_sub = rates[:, :, 1]
        lam_del = rates[:, :, 2]

        ux_ins = lam_ins.unsqueeze(-1) * ins_probs
        ux_sub = lam_sub.unsqueeze(-1) * sub_probs
        ux_del = lam_del.unsqueeze(-1)

        ux_cat = torch.cat([ux_ins, ux_sub, ux_del], dim=-1)

        uz_cat = fill_gap_tokens_with_repeats(ux_cat, z_gap, z_pad)

        uz_mask = make_ut_mask_from_z(zt, z1, vocab_size=BASE_VOCAB + 2)

        u_tot = rates.sum(dim=(1, 2))

        sched_coeff = (self.kappa.derivative(t) / (1 - self.kappa(t))).to(self.device)

        log_uz_cat = torch.clamp(uz_cat.log(), min=-20)

        loss_vec = u_tot - (log_uz_cat * uz_mask * sched_coeff.unsqueeze(-1)).sum(dim=(1, 2))

        return loss_vec.mean(), {
            'u_tot': u_tot.mean(),
            'u_ins': lam_ins.sum(1).mean(),
            'u_sub': lam_sub.sum(1).mean(),
            'u_del': lam_del.sum(1).mean(),
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

    @torch.no_grad()
    def sample(self, x0, n_steps=100, t_min=0.0):
        self.model.eval()
        device = x0.device
        t = torch.full((x0.size(0), 1), t_min, device=device)
        dt = 1 / n_steps
        pad_mask = (x0 == PAD_TOKEN)
        xt = x0.clone()
        traj = [xt.clone()]
        for _ in range(n_steps):
            rates, ins_probs, sub_probs = self(xt, t, pad_mask)
            lam_i, lam_s, lam_d = rates[..., 0], rates[..., 1], rates[..., 2]

            ins_mask = torch.rand_like(lam_i) < 1 - torch.exp(-dt * lam_i)
            ds_mask = torch.rand_like(lam_s) < 1 - torch.exp(-dt * (lam_s + lam_d))

            prob_del = torch.where(ds_mask, lam_d / (lam_s + lam_d + 1e-8), torch.zeros_like(lam_d))

            del_mask = torch.bernoulli(prob_del).bool()
            sub_mask = ds_mask & ~del_mask
            non_pad = ~pad_mask

            ins_tokens = torch.full_like(xt, PAD_TOKEN)
            sub_tokens = torch.full_like(xt, PAD_TOKEN)

            if non_pad.any():
                ins_tokens[non_pad] = torch.multinomial(ins_probs[non_pad], 1).squeeze(-1)
                sub_tokens[non_pad] = torch.multinomial(sub_probs[non_pad], 1).squeeze(-1)

            xt[sub_mask] = sub_tokens[sub_mask]
            # simplified insertion/deletion (no complex shifting here)
            xt[del_mask] = PAD_TOKEN
            # naive insert: replace first PAD position after site

            for b in range(xt.size(0)):
                positions = torch.nonzero(ins_mask[b], as_tuple=True)[0]
                for p in positions.tolist():
                    pad_pos = (xt[b] == PAD_TOKEN).nonzero(as_tuple=True)[0]
                    if pad_pos.numel():
                        tgt = pad_pos[0].item()
                        xt[b, tgt] = ins_tokens[b, p]

            pad_mask = (xt == PAD_TOKEN)
            t = t + dt
            traj.append(xt.clone())

        return traj
