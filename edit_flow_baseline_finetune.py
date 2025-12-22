import math
import os
import typing
import pickle

import hydra.utils
import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor
from tqdm import tqdm

import dataloader
import models
from edit_flow_baseline import EditFlowBaseline, fill_gap_tokens_with_repeats, make_uz_mask, apply_ins_del_operations
from flow_utils import rm_gap_tokens
from flows import CubicScheduler, x2prob, sample_p


class EditFlowBaselineFineTune(EditFlowBaseline):
    def __init__(
            self,
            config,
            tokenizer: transformers.PreTrainedTokenizer):
        super().__init__(config, tokenizer)

    def _compute_loss(self, batch):
        x_0, x_1, z_0, z_1, t, context_mask = batch

        z_t = self.sample_zt_sparse(z_0, z_1, t)

        x_t, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z_t, pad_token=self.pad_token, gap_token=self.gap_token)

        x_t = torch.where(context_mask, x_1, x_t)

        u_t, sub_logits, ins_logits = self.backbone.forward(x_t, t, x_pad_mask)

        mask_expanded = context_mask.unsqueeze(-1).bool()
        u_t = u_t.masked_fill(mask_expanded, 0.0)
        # Force padded logits to -Inf (so softmax=0)
        sub_logits = sub_logits.masked_fill(mask_expanded, -1e9)
        ins_logits = ins_logits.masked_fill(mask_expanded, -1e9)

        u_tot = u_t.sum(dim=(1, 2))

        eps = 1e-9

        log_rates = torch.log(u_t + eps)

        lse_sub_x = sub_logits.logsumexp(dim=-1, keepdim=True)  # [B, Sx, 1]
        lse_ins_x = ins_logits.logsumexp(dim=-1, keepdim=True)  # [B, Sx, 1]

        packed_features_x = torch.cat([log_rates, lse_sub_x, lse_ins_x], dim=-1)

        packed_features_z = fill_gap_tokens_with_repeats(
            packed_features_x, z_gap_mask, z_pad_mask
        )

        log_rate_ins = packed_features_z[..., 0]
        log_rate_sub = packed_features_z[..., 1]
        log_rate_del = packed_features_z[..., 2]
        lse_sub_z = packed_features_z[..., 3]
        lse_ins_z = packed_features_z[..., 4]

        non_gap_mask = ~z_gap_mask
        x_indices = non_gap_mask.cumsum(dim=1) - 1
        x_indices = x_indices.clamp(min=0, max=x_t.shape[1] - 1)

        valid_vocab_limit = sub_logits.size(-1) - 1
        safe_z1 = z_1.clamp(min=0, max=valid_vocab_limit)
        batch_idx = torch.arange(x_t.shape[0], device=self.device).unsqueeze(1)

        target_sub_logits = sub_logits[batch_idx, x_indices, safe_z1]
        target_ins_logits = ins_logits[batch_idx, x_indices, safe_z1]

        uz_mask = make_uz_mask(z_t, z_1, self.pad_token, self.gap_token)

        target_sub_mask = uz_mask[:, :, 0] & ~context_mask
        target_ins_mask = uz_mask[:, :, 1] & ~context_mask
        target_del_mask = uz_mask[:, :, 2] & ~context_mask

        term_ins = (log_rate_ins + target_ins_logits - lse_ins_z) * target_ins_mask
        term_del = (log_rate_del) * target_del_mask
        term_sub = (log_rate_sub + target_sub_logits - lse_sub_z) * target_sub_mask

        selected_log_ll = term_ins + term_del + term_sub

        default_coeff = (self.default_scheduler.derivative(t) / (1 - self.default_scheduler(t) + eps))  # .squeeze()

        term2 = (selected_log_ll * default_coeff).sum(dim=1)

        loss = u_tot - term2

        u_ins = u_t[:, :, 0].sum(dim=1).mean()
        u_del = u_t[:, :, 2].sum(dim=1).mean()
        u_sub = u_t[:, :, 1].sum(dim=1).mean()

        return loss.mean(), u_tot, u_ins, u_del, u_sub, term2.mean()

    def on_validation_epoch_start(self):
        self.backbone.eval()

    # todo setup validation steps for conditional sampling, set up dataloader/dataset
    def validation_step(self, batch, batch_idx):
        loss, u_tot, u_ins, u_del, u_sub, term2 = self._compute_loss(batch)
        self.log_dict(
            {
                "val_loss": loss,
                "val_u_tot": u_tot.mean(),
                "val_u_ins": u_ins,
                "val_u_del": u_del,
                "val_u_sub": u_sub,
                "val_term2": term2,
            }, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        if ((self.config.eval.compute_perplexity_on_sanity
             or not self.trainer.sanity_checking)
                and self.config.eval.generate_samples):

            samples, text_samples = None, None
            for _ in range(self.config.sampling.num_sample_batches):
                samples = self._sample()

                # Decode the samples to be re-tokenized by eval model
                text_samples = self.tokenizer.batch_decode(samples)

                if self.config.eval.compute_generative_perplexity:
                    self.compute_generative_perplexity(text_samples)

            if self.trainer.global_rank == 0 and hasattr(self.trainer.logger, 'log_table'):
                # Log the last generated samples
                text_samples = text_samples[:self.config.sampling.num_sample_log]

                self.trainer.logger.log_table(
                    key=f'samples@global_step{self.global_step}',
                    columns=['Generated Samples'],
                    data=[[s] for s in text_samples])

            if self.config.eval.compute_generative_perplexity:
                self.log('val/gen_ppl',
                         self.gen_ppl_metric,
                         on_epoch=True,
                         on_step=False,
                         sync_dist=True,
                         prog_bar=True)

    @torch.no_grad()
    def _sample_conditional(self, x_0, context_mask, n_steps=None, eps=1e-5):
        batch_size_per_gpu = self.config.loader.eval_batch_size

        # Lightning auto-casting is not working in this method for some reason
        if n_steps is None:
            n_steps = self.config.sampling.steps

        default_h = 1 / n_steps

        t_min = 0.01

        t = t_min * torch.ones(batch_size_per_gpu, 1, device=self.device)

        x_t = x_0.clone().to(self.device)

        x_pad_mask = (x_t == self.pad_token)  # Create padding mask for x_t
        # x_ts = [x_t.clone()]

        with tqdm(desc="Euler Sampling") as pbar:
            while t.max() <= 1:
                u_t, sub_logits, ins_logits = self.backbone.forward(x_t, t, x_pad_mask)

                mask_expanded = context_mask.unsqueeze(-1).bool()
                u_t = u_t.masked_fill(mask_expanded, 0.0)
                # Force padded logits to -Inf (so softmax=0)
                sub_logits = sub_logits.masked_fill(mask_expanded, -1e9)
                ins_logits = ins_logits.masked_fill(mask_expanded, -1e9)

                sub_probs = F.softmax(sub_logits, dim=-1)
                ins_probs = F.softmax(ins_logits, dim=-1)

                valid_token_mask = (~x_pad_mask & ~context_mask).float()

                lambda_ins = u_t[:, :, 0] * valid_token_mask
                lambda_sub = u_t[:, :, 1] * valid_token_mask
                lambda_del = u_t[:, :, 2] * valid_token_mask

                adapt_h = default_h

                # Sample insertions and deletion/substitutions based on rates
                ins_mask = torch.rand(
                    size=lambda_ins.shape, device=lambda_ins.device) < 1 - torch.exp(-adapt_h * lambda_ins)
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
                ins_tokens = torch.full(ins_probs.shape[:2], self.pad_token, dtype=torch.long, device=self.device)
                sub_tokens = torch.full(sub_probs.shape[:2], self.pad_token, dtype=torch.long, device=self.device)

                non_pad_mask = ~x_pad_mask

                if non_pad_mask.any():
                    ins_sampled = torch.multinomial(ins_probs[non_pad_mask], num_samples=1, replacement=True).squeeze(
                        -1)
                    sub_sampled = torch.multinomial(sub_probs[non_pad_mask], num_samples=1, replacement=True).squeeze(
                        -1)
                    ins_tokens[non_pad_mask] = ins_sampled
                    sub_tokens[non_pad_mask] = sub_sampled

                # Apply operations based on masks
                x_t[sub_mask] = sub_tokens[sub_mask]
                x_t = apply_ins_del_operations(
                    x_t,
                    ins_mask,
                    del_mask,
                    ins_tokens,
                    max_seq_len=self.config.model.length,
                    pad_token=self.pad_token,
                )

                x_pad_mask = (x_t == self.pad_token)  # Update padding mask after operations

                t = t + adapt_h
                x_t = torch.where(context_mask, x_0, x_t)

                # x_ts.append(x_t.clone())
                pbar.update(1)

        return x_t
