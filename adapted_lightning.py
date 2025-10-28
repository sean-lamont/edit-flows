import lightning.pytorch as pl
from sacrebleu.metrics import BLEU
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
from torch.nn import functional as F
from torch.optim import Optimizer
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import torch
import deepspeed
import tqdm
import wandb

from scheduler import CubicScheduler
from utils import *
from lightning.pytorch.utilities import grad_norm
# import torchviz



class AdaptedLitModule(pl.LightningModule):
    def __init__(self, model: nn.Module, full_vocab_size, pad_token_id, gap_token_id, lr=1e-4, scheduler_cfg=None,
                 anneal_end_step=10000):
        super().__init__()
        # self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.kappa = CubicScheduler(**(scheduler_cfg or {'a': 0.0, 'b': 2.0})) # from discrete flow matching paper, =t^2
        self.anneal_end_step = anneal_end_step
        self.full_vocab_size = full_vocab_size
        self.pad_token = pad_token_id
        self.gap_token = gap_token_id
        self.lr = lr
        self.tokenizer = AutoTokenizer.from_pretrained("Goedel-LM/Goedel-Prover-V2-8B")
        # big hack: use a dummy model with small mem footprint to recover from OOM  mid training
        # clear cache if OOM in forward, run dummy model as new loss,
        self.dummy_model = torch.nn.Linear(1, 1, dtype=torch.bfloat16)
        self.val_sample_count = 5
        self.max_seq_len = 7000
        self.oom_count = 0
        self.bleu = BLEU()
        self.val_t = None



    def configure_optimizers(self):
        return DeepSpeedCPUAdam(self.parameters(), lr=self.lr, betas=(0.9, 0.95))
        # return DeepSpeedCPUAdam(self.parameters(), 1e-5, eps=1e-6)
        return torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.95))
        # opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.95))
        # steps = 500000 # change based on total steps
        # scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=2000, num_training_steps=steps)
        # return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}

    def on_validation_start(self):
        self.val_outputs = wandb.Table(
            columns=["global_step", "epoch", "Initial Seq", "Final Seq" ],
            log_mode="INCREMENTAL"
        )


    def forward(self, tokens, t, pad_mask, contexts,  attn_mask_ratio):
        return self.model(tokens, t, pad_mask, contexts, self.pad_token, attn_mask_ratio)

    def _loss(self, batch):
        x1, x0, z0, z1, t, context_lens, contexts = batch['x1'], batch['x0'], batch['z0'], batch['z1'], batch['t'], batch[
            'context_lens'], batch['contexts']

        p0 = x2prob(z0, self.full_vocab_size)
        p1 = x2prob(z1, self.full_vocab_size)

        zt = sample_cond_pt(p0, p1, t, self.kappa)

        xt, x_pad, z_gap, z_pad = rm_gap_tokens(zt, self.pad_token, self.gap_token)

        attn_mask_ratio = min(1.0, self.global_step / self.anneal_end_step)

        rates, ins_probs, sub_probs = self.forward(xt, t, x_pad, contexts, attn_mask_ratio, )

        lam_ins = rates[:, :, 0]
        lam_sub = rates[:, :, 1]
        lam_del = rates[:, :, 2]

        ux_ins = lam_ins.unsqueeze(-1) * ins_probs
        ux_sub = lam_sub.unsqueeze(-1) * sub_probs
        ux_del = lam_del.unsqueeze(-1)

        ux_cat = torch.cat([ux_ins, ux_sub, ux_del], dim=-1)

        uz_cat = fill_gap_tokens_with_repeats(ux_cat, z_gap, z_pad)

        # which operations move closer to z_1 from z_t
        # (batch_size, z_seq_len, 2 * vocab_size)
        uz_mask = make_ut_mask_from_z(zt, z1, self.full_vocab_size, self.pad_token, self.gap_token)


        u_tot = rates.sum(dim=(1, 2))

        sched_coeff = (self.kappa.derivative(t) / (1 - self.kappa(t))).to(self.device)

        log_uz_cat = torch.clamp(uz_cat.log(), min=-20)

        term2  = (log_uz_cat * uz_mask * sched_coeff.unsqueeze(-1)).sum(dim=(1, 2))

        loss_vec = u_tot - term2

        # normalise by aligned sequence length
        N = torch.sum(~z_pad, dim=1).float()

        N = torch.clamp(N, min=1.0)

        loss_vec = loss_vec / N

        if torch.isnan(loss_vec).any():
            print (f'nan loss')
            return self.dummy_loss(batch)

        
        # self.training_step_outputs.add_data(
        #     self.global_step,
        #     self.current_epoch,
        #     self.tokenizer.batch_decode(x1),
        #     self.tokenizer.batch_decode(xt),
        #     self.tokenizer.batch_decode(z1),
        #     self.tokenizer.batch_decode(xt),
        # )

        # self.logger.experiment.log(
        #     {"training_step_outputs": self.training_step_outputs},
        #     step=self.global_step
        # )

        # todo could do nan check here and run dummy model

        return loss_vec.mean(), {
            'utot': u_tot.mean(),
            'u_tot / N': (u_tot / N).mean(),
            '-term2': -term2.mean(),
            '-term2 / N': -(term2 / N).mean(),
            'u_ins': lam_ins.sum(1).mean(),
            'u_sub': lam_sub.sum(1).mean(),
            'u_del': lam_del.sum(1).mean(),
            'attn_mask_ratio': attn_mask_ratio,
            'N': N,
            't': t,
            'idx': batch['idx'][0],
            'edit_dist': uz_mask.sum().float()
        }

    def dummy_loss(self, batch):
        return torch.sum(self.dummy_model(torch.ones(batch['x0'].shape[0], device=self.device, dtype=torch.bfloat16)),
                         dim=-1)

    def training_step(self, batch, batch_idx):
        try:
            loss, metrics = self._loss(batch)
            self.log('train/loss', loss, prog_bar=True)
            for k, v in metrics.items():
                self.log(f'train/{k}', v, prog_bar=False)
            return loss
        except Exception as e:
            if 'CUDA out of memory' in str(e):
                self.oom_count = self.oom_count + 1
                self.log(f'train/oom_count', self.oom_count, prog_bar=False)
                torch.cuda.empty_cache()
                return self.dummy_loss(batch)
            else:
                print (f'non OOM error: {e}')
                torch.cuda.empty_cache()
                return self.dummy_loss(batch)

    def validation_step(self, batch, batch_idx):
        try:
            # use fixed t for more consistent results
            if not self.val_t:
                self.val_t = torch.rand(batch['t'].shape[0], 1, device=batch['t'].device)
                self.val_t = torch.clamp(self.val_t - 1e-2, min=0.0) # subtract eps to account for occasional 1's

            batch['t'] = self.val_t

            loss, metrics = self._loss(batch)
            self.log('val/loss', loss, prog_bar=True)
            for k, v in metrics.items():
                self.log(f'val/{k}', v, prog_bar=False)

            # sample first group of batches, rather than random, for better comparisons over training
            # only take first element in batch
            if batch_idx < self.val_sample_count:
                x0_sample = batch['x0'][0].unsqueeze(0)
                context = [batch['contexts'][0]]

                # Generate a trajectory
                trajectory = self.sample(x0_sample, context, n_steps=100)

                # Log the initial and final states of the trajectory
                initial_seq = self.tokenizer.decode(trajectory[0].squeeze().tolist(), skip_special_tokens=False)
                final_seq = self.tokenizer.decode(trajectory[-1].squeeze().tolist(), skip_special_tokens=False)

                # add bleu score comparison between final_seq and target
                # Calculate BLEU score if a target sequence is available in the batch
                if 'target_seq' in batch:
                    target_seq = self.tokenizer.decode(batch['target_seq'][0].squeeze().tolist(), skip_special_tokens=False)
                    # Assuming target_seq is a string and final_seq is a string
                    # You might need to tokenize them into lists of words for sacrebleu
                    score = self.bleu.corpus_score([final_seq], [[target_seq]]).score
                    self.log('val/bleu_score', score, prog_bar=True)


                self.val_outputs.add_data(
                    self.global_step,
                    self.current_epoch,
                    initial_seq,
                    final_seq
                )

                self.logger.experiment.log(
                    {"val_outputs": self.val_outputs},
                    step=self.global_step
                )

                # # Optionally, log the full trajectory as a list of strings
                # full_trajectory_decoded = [self.tokenizer.decode(seq.squeeze().tolist(), skip_special_tokens=False) for seq in trajectory]
                # self.logger.experiment.log({
                #     f"val/sample_{i}/full_trajectory": wandb.Table(data=[[s] for s in full_trajectory_decoded], columns=["sequence"])
                # })

        except Exception as e:
            if 'CUDA out of memory' in str(e):
                self.oom_count = self.oom_count + 1
                self.log(f'val/oom_count', self.oom_count, prog_bar=False)
                torch.cuda.empty_cache()
                return

            else:
                print(f'Non OOM error in val: {e}')
                torch.cuda.empty_cache()
                return

    @torch.no_grad()
    def sample(self, x0, context, n_steps=100, t_min=0.0):
        self.model.eval()
        device = x0.device
        t = torch.full((x0.size(0), 1), t_min, device=device, dtype=torch.bfloat16)
        dt = 1 / n_steps
        pad_mask = (x0 == self.pad_token)
        xt = x0.clone()
        traj = [xt.clone()]

        for _ in range(n_steps):
            attn_mask_ratio = min(1.0, self.global_step / self.anneal_end_step)

            adapt_h = get_adaptive_h(dt, t, self.kappa)
            # adapt_h = dt

            # print (f'dtype: {xt.dtype, t.dtype, pad_mask.dtype}')
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

            if non_pad.any():
                ins_tokens[non_pad] = torch.multinomial(ins_probs[non_pad], 1).squeeze(-1)
                sub_tokens[non_pad] = torch.multinomial(sub_probs[non_pad], 1).squeeze(-1)

            xt[sub_mask] = sub_tokens[sub_mask]

            xt = apply_ins_del(xt, ins_mask, del_mask, ins_tokens, max_seq_len=self.max_seq_len, pad_token=self.pad_token)

            pad_mask = (xt == self.pad_token)
            t = t + adapt_h
            traj.append(xt.clone())

        return traj


def apply_ins_del(xt, ins_mask, del_mask, ins_tokens, max_seq_len, pad_token):
        """
        Apply insertion and deletion operations to a sequence x_t based on the provided masks.
        """
        batch_size, seq_len = xt.shape
        device = xt.device

        # Handle simultaneous ins+del as substitutions
        replace_mask = ins_mask & del_mask
        x_t_modified = xt.clone()
        x_t_modified[replace_mask] = ins_tokens[replace_mask]

        # Update ins/del masks after handling replacements
        eff_ins_mask = ins_mask & ~replace_mask
        eff_del_mask = del_mask & ~replace_mask

        # Compute new lengths after applying ins/del operations
        xt_pad_mask = (xt == pad_token)  # (batch_size, seq_len)
        xt_seq_lens = (~xt_pad_mask).sum(dim=1)  # (batch_size,)
        new_lengths = xt_seq_lens + eff_ins_mask.sum(dim=1) - eff_del_mask.sum(dim=1)
        max_new_len = int(new_lengths.max().item())

        if max_new_len <= 0:
            print(f"Unexpected max_new_len <= 0: {max_new_len}, did we delete everything?")
            return torch.full((batch_size, 1), pad_token, dtype=xt.dtype, device=device),

        # Pre-allocate result
        x_new = torch.full((batch_size, max_new_len), pad_token, dtype=xt.dtype, device=device)

        # Compute positions
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)  # (batch_size, 1)
        pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
        cum_del = torch.cumsum(eff_del_mask.float(), dim=1)  # num del up to & incl. current pos
        cum_ins = torch.cumsum(eff_ins_mask.float(), dim=1)  # num ins up to & incl. current pos
        cum_ins_before = F.pad(cum_ins[:, :-1], (1, 0), value=0)  # num ins before current pos

        # Place non-deleted tokens
        new_pos = pos_idx + cum_ins_before - cum_del  # new pos of tokens shifted by ins/del
        keep_mask = ~eff_del_mask & (new_pos >= 0) & (new_pos < max_new_len)  # tokens to keep (non-deleted)
        if keep_mask.any():
            x_new[batch_idx.expand(-1, seq_len)[keep_mask], new_pos[keep_mask].long()] = x_t_modified[keep_mask]

        # Place insertions
        if eff_ins_mask.any():
            ins_pos = new_pos + 1  # insertions go 1 after new shifted pos
            ins_valid = eff_ins_mask & (ins_pos >= 0) & (ins_pos < max_new_len)  # tokens to insert
            if ins_valid.any():
                x_new[batch_idx.expand(-1, seq_len)[ins_valid], ins_pos[ins_valid].long()] = ins_tokens[ins_valid]

        if max_new_len > max_seq_len:
            print(f"Warning: max_new_len {max_new_len} exceeds max_seq_len {max_seq_len}, truncating.")
            max_new_len = max_seq_len

        return x_new[:, :max_new_len]

def get_adaptive_h(h: float, t: torch.Tensor, scheduler):
    coeff = (1 - scheduler(t)) / scheduler.derivative(t)
    _h = h * torch.ones_like(t, device=t.device)
    h_adapt = torch.minimum(_h, coeff)
    return h_adapt