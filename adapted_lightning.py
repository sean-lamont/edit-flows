import bitsandbytes.optim as bnb_optim
import lightning.pytorch as pl
from sacrebleu.metrics import BLEU
from torch.nn import functional as F

import wandb
from scheduler import CubicScheduler
from utils import *


def top_p_probs(probs, top_p):
    # Sort the probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Calculate cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find the index where the cumulative probability exceeds top_p
    sorted_indices_to_remove = cumulative_probs > top_p

    # Ensure at least one token is selected (take the highest prob)
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = torch.tensor(False, device=cumulative_probs.device)

    # Scatter back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

    # Set probabilities of tokens to remove to 0
    probs[indices_to_remove] = torch.tensor(0.0, device=probs.device)

    # Renormalize the probabilities
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs

class AdaptedLitModule(pl.LightningModule):
    def __init__(self, model: nn.Module,  tokenizer, pad_token_id, gap_token_id, lr=1e-4, scheduler_cfg=None,
                 anneal_end_step=10000):
        super().__init__()
        self.model = model
        self.kappa = CubicScheduler(**(scheduler_cfg or {'a': 0.0, 'b': 2.0}))
        self.anneal_end_step = anneal_end_step
        self.tokenizer = tokenizer
        self.pad_token = pad_token_id
        self.gap_token = gap_token_id
        self.lr = lr
        self.val_sample_count = 5
        self.max_seq_len = 7000
        self.bleu = BLEU()
        self.val_t = None
        # (configure_optimizers and on_validation_epoch_start are unchanged)

    def configure_optimizers(self):
        return bnb_optim.AdamW8bit(self.parameters(), lr=self.lr, betas=(0.9, 0.95), percentile_clipping=5, weight_decay=1e-2, eps=1e-6)

    def on_validation_epoch_start(self):
        self.val_outputs = wandb.Table(
            columns=["global_step", "epoch", "Initial Seq", "Final Seq", "Target Seq"],
            log_mode="INCREMENTAL"
        )

    def forward(self, tokens, t, pad_mask, contexts, attn_mask_ratio):
        # This function call is unchanged, but we now pass `xt_pad` to the `pad_mask` argument
        return self.model(tokens, t, pad_mask, contexts, self.pad_token, attn_mask_ratio)

    def _loss(self, batch):
        xt = batch['xt']
        contexts = batch['contexts']  # This is a list[Tensor]
        t = batch['t']
        uz_mask = batch['uz_mask']
        xt_pad = batch['xt_pad']  # Padding mask for xt
        z_gap = batch['z_gap']  # Gap mask from zt
        z_pad = batch['z_pad']  # Padding mask from z0/z1

        attn_mask_ratio = min(1.0, self.global_step / self.anneal_end_step)

        rates, ins_probs, sub_probs = self.forward(xt, t, xt_pad, contexts, attn_mask_ratio, )

        lam_ins = rates[:, :, 0]
        lam_sub = rates[:, :, 1]
        lam_del = rates[:, :, 2]

        ux_ins = lam_ins.unsqueeze(-1) * ins_probs
        ux_sub = lam_sub.unsqueeze(-1) * sub_probs
        ux_del = lam_del.unsqueeze(-1)

        ux_cat = torch.cat([ux_ins, ux_sub, ux_del], dim=-1)

        uz_cat = fill_gap_tokens_with_repeats(ux_cat, z_gap, z_pad)

        u_tot = rates.sum(dim=(1, 2))

        sched_coeff = (self.kappa.derivative(t) / (1 - self.kappa(t))).to(self.device)

        log_uz_cat = torch.clamp(uz_cat.log(), min=-20)

        term2 = (log_uz_cat * uz_mask * sched_coeff.unsqueeze(-1)).sum(dim=(1, 2))

        loss_vec = u_tot - term2

        N = torch.sum(~z_pad, dim=1).float()
        N = torch.clamp(N, min=1.0)

        loss_vec = loss_vec  # / N

        return loss_vec.mean(), {
            'utot': u_tot.mean(),
            'u_tot / N': (u_tot / N).mean(),
            '-term2': -term2.mean(),
            '-term2 / N': -(term2 / N).mean(),
            'u_ins': lam_ins.sum(1).mean(),
            'u_sub': lam_sub.sum(1).mean(),
            'u_del': lam_del.sum(1).mean(),
            'attn_mask_ratio': attn_mask_ratio,
            'N': N.mean(),
            't': t.mean(),
            'edit_dist': uz_mask.sum().float()
        }

    def training_step(self, batch, batch_idx):
        loss, metrics = self._loss(batch)
        self.log('train/loss', loss, prog_bar=True, on_epoch=True)
        for k, v in metrics.items():
            self.log(f'train/{k}', v, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # use fixed t for more consistent results
        if self.val_t is None:
            self.val_t = torch.rand(batch['t'].shape[0], 1, device=batch['t'].device, dtype=torch.bfloat16)
            self.val_t = torch.clamp(self.val_t - 1e-2, min=0.0)

        batch['t'] = self.val_t

        loss, metrics = self._loss(batch)
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        for k, v in metrics.items():
            self.log(f'val/{k}', v, prog_bar=False, sync_dist=True)

        # if batch_idx < self.val_sample_count or self.val_sample_count == -1:
        if True: # todo until we load from just lora weights
            # --- MODIFIED: Use x0_padded and x1_padded from batch ---
            x0_sample = batch['x0_padded'][0].unsqueeze(0)
            context = [batch['contexts'][0]]
            target_seq = self.tokenizer.decode(batch['x1_padded'][0].squeeze().tolist(), skip_special_tokens=False)

            # generate a trajectory
            trajectory = self.sample(x0_sample, context, n_steps=1000)

            # log the initial and final states of the trajectory
            initial_seq = self.tokenizer.decode(trajectory[0].squeeze().tolist(), skip_special_tokens=False)
            final_seq = self.tokenizer.decode(trajectory[-1].squeeze().tolist(), skip_special_tokens=False)

            score = self.bleu.corpus_score([final_seq], [[target_seq]]).score
            self.log('val/bleu_score', score, prog_bar=True, sync_dist=True)

            self.val_outputs.add_data(
                self.global_step,
                self.current_epoch,
                initial_seq,
                final_seq,
                target_seq
            )

    def on_validation_epoch_end(self) -> None:
        self.logger.experiment.log(
            {"val_outputs": self.val_outputs}
        )

    @torch.no_grad()
    def sample(self, x0, context, n_steps=100, t_min=0.0, top_p=0.5):
        # (This function is unchanged)
        self.model.eval()
        device = x0.device
        t = torch.full((x0.size(0), 1), t_min, device=device, dtype=torch.bfloat16)
        dt = 1 / n_steps
        pad_mask = (x0 == self.pad_token)
        xt = x0.clone()
        traj = [xt.clone()]

        for _ in range(n_steps):
            attn_mask_ratio = min(1.0, self.global_step / self.anneal_end_step) # todo make this a class param updated during training only
            # print (attn_mask_ratio)
            attn_mask_ratio = 1.0

            adapt_h = get_adaptive_h(dt, t, self.kappa)
            rates, ins_probs, sub_probs = self.forward(xt, t, pad_mask, context, attn_mask_ratio)

            lam_i, lam_s, lam_d = rates[..., 0], rates[..., 1], rates[..., 2]

            ins_mask = torch.rand_like(lam_i) < 1 - torch.exp(-adapt_h * lam_i)
            ds_mask = torch.rand_like(lam_s) < 1 - torch.exp(-adapt_h * (lam_s + lam_d))

            prob_del = torch.where(ds_mask, lam_d / (lam_s + lam_d + 1e-8), torch.zeros_like(lam_d))
            del_mask = torch.bernoulli(prob_del).bool()
            sub_mask = ds_mask & ~del_mask
            non_pad = ~pad_mask

            ins_tokens = torch.full_like(xt, self.pad_token)
            sub_tokens = torch.full_like(xt, self.pad_token)

            if top_p < 1.0:
                # Apply top-p sampling
                ins_probs = top_p_probs(ins_probs[non_pad], top_p)
                sub_probs = top_p_probs(sub_probs[non_pad], top_p)

            if non_pad.any():
                ins_tokens[non_pad] = torch.multinomial(ins_probs, 1).squeeze(-1)
                sub_tokens[non_pad] = torch.multinomial(sub_probs, 1).squeeze(-1)

            xt[sub_mask] = sub_tokens[sub_mask]
            xt = apply_ins_del(xt, ins_mask, del_mask, ins_tokens, max_seq_len=self.max_seq_len,
                               pad_token=self.pad_token)
            pad_mask = (xt == self.pad_token)
            t = t + adapt_h
            traj.append(xt.clone())

        return traj


def apply_ins_del(xt, ins_mask, del_mask, ins_tokens, max_seq_len, pad_token):
    batch_size, seq_len = xt.shape
    device = xt.device
    replace_mask = ins_mask & del_mask
    x_t_modified = xt.clone()
    x_t_modified[replace_mask] = ins_tokens[replace_mask]
    eff_ins_mask = ins_mask & ~replace_mask
    eff_del_mask = del_mask & ~replace_mask
    xt_pad_mask = (xt == pad_token)
    xt_seq_lens = (~xt_pad_mask).sum(dim=1)
    new_lengths = xt_seq_lens + eff_ins_mask.sum(dim=1) - eff_del_mask.sum(dim=1)
    max_new_len = int(new_lengths.max().item())
    if max_new_len <= 0:
        return torch.full((batch_size, 1), pad_token, dtype=xt.dtype, device=device),
    x_new = torch.full((batch_size, max_new_len), pad_token, dtype=xt.dtype, device=device)
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)
    pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)
    cum_del = torch.cumsum(eff_del_mask.float(), dim=1)
    cum_ins = torch.cumsum(eff_ins_mask.float(), dim=1)
    cum_ins_before = F.pad(cum_ins[:, :-1], (1, 0), value=0)
    new_pos = pos_idx + cum_ins_before - cum_del
    keep_mask = ~eff_del_mask & (new_pos >= 0) & (new_pos < max_new_len)
    if keep_mask.any():
        x_new[batch_idx.expand(-1, seq_len)[keep_mask], new_pos[keep_mask].long()] = x_t_modified[keep_mask]
    if eff_ins_mask.any():
        ins_pos = new_pos + 1
        ins_valid = eff_ins_mask & (ins_pos >= 0) & (ins_pos < max_new_len)
        if ins_valid.any():
            x_new[batch_idx.expand(-1, seq_len)[ins_valid], ins_pos[ins_valid].long()] = ins_tokens[ins_valid]
    if max_new_len > max_seq_len:
        max_new_len = max_seq_len
    return x_new[:, :max_new_len]


def get_adaptive_h(h: float, t: torch.Tensor, scheduler):
    coeff = (1 - scheduler(t)) / scheduler.derivative(t)
    _h = h * torch.ones_like(t, device=t.device)
    h_adapt = torch.minimum(_h, coeff)
    return h_adapt