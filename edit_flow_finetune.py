import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

from edit_flow import EditFlow, poisson_make_uz_mask, fill_gap_tokens_with_repeats
from flow_utils import rm_gap_tokens


class EditFlowFineTune(EditFlow):
    def __init__(
            self,
            config,
            tokenizer: transformers.PreTrainedTokenizer):
        super().__init__(config, tokenizer)

    def _compute_loss(self, batch):
        # context mask = (bsz, seq_len) where we keep original context (i.e. x0, z0, z1, x1 should all have context_ids the same)
        x_0, x_1, z_0, z_1, t, context_mask = batch

        z_t = self.sample_zt_sparse(z_0, z_1, t)

        x_t, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z_t, pad_token=self.pad_token, gap_token=self.gap_token)

        # we set x_t to x_1 for context_ids (assumes x0, x1, z0, z1 all have context at same ids!)

        x_t = torch.where(context_mask, x_1, x_t)

        u_t, sub_logits = self.backbone.forward(x_t, t, x_pad_mask)

        mask_expanded = context_mask.unsqueeze(-1).bool()
        u_t = u_t.masked_fill(mask_expanded, 0.0)
        # Force padded logits to -Inf (so softmax=0)
        sub_logits = sub_logits.masked_fill(mask_expanded, -1e9)

        u_tot = u_t.sum(dim=(1, 2))

        uz_mask = poisson_make_uz_mask(z_t, z_1, vocab_size=self.V,
                                       gap_token=self.gap_token, pad_token=self.pad_token)

        target_sub = uz_mask[:, :, 0] & ~context_mask
        target_ins = uz_mask[:, :, 1] & ~context_mask
        target_del = uz_mask[:, :, 2] & ~context_mask

        log_sum_exp_x = sub_logits.logsumexp(dim=-1)

        log_sum_exp_z = fill_gap_tokens_with_repeats(
            log_sum_exp_x.unsqueeze(-1), z_gap_mask, z_pad_mask
        ).squeeze(-1)

        non_gap_mask = ~z_gap_mask
        x_indices = non_gap_mask.cumsum(dim=1) - 1
        x_indices = x_indices.clamp(min=0, max=x_t.shape[1] - 1)

        valid_vocab_limit = sub_logits.size(-1) - 1
        safe_z1 = z_1.clamp(min=0, max=valid_vocab_limit)

        batch_idx = torch.arange(x_t.shape[0], device=self.device).unsqueeze(1)

        target_logits_z = sub_logits[batch_idx, x_indices, safe_z1]

        vocab_nll = log_sum_exp_z - target_logits_z

        eps = 1e-9
        log_rates = torch.log(u_t + eps)
        uz_log_rates = fill_gap_tokens_with_repeats(log_rates, z_gap_mask, z_pad_mask)

        log_rate_ins = uz_log_rates[:, :, 0]
        log_rate_sub = uz_log_rates[:, :, 1]
        log_rate_del = uz_log_rates[:, :, 2]

        selected_log_ll = (
                (log_rate_ins * target_ins) +
                (log_rate_del * target_del) +
                ((log_rate_sub - vocab_nll) * target_sub)
        )

        default_coeff = (self.default_scheduler.derivative(t) / (1 - self.default_scheduler(t) + eps))
        ins_coeff = (self.mask_scheduler.derivative(t) / (1 - self.mask_scheduler(t) + eps))

        mask_sub_coeff = (self.mask_scheduler.derivative(t) * self.default_scheduler(t)
                          + self.mask_scheduler(t) * self.default_scheduler.derivative(t)) / (
                                 self.mask_scheduler(t) * (1 - self.default_scheduler(t)) + eps)

        z_ins_event = (z_0 == self.gap_token) & (z_1 != self.gap_token) & (z_0 != z_1)

        mask_ids = (z_t == self.mask_token) & (z_1 != self.mask_token) & (z_0 != self.mask_token) & (z_1 != z_0)

        sched_coeff = torch.where(z_ins_event, ins_coeff, default_coeff)

        sched_coeff = torch.where(mask_ids, mask_sub_coeff, sched_coeff)

        term2 = (selected_log_ll * sched_coeff).sum(dim=1)
        loss = u_tot - term2

        u_ins = u_t[:, :, 0].sum(dim=1).mean()
        u_del = u_t[:, :, 2].sum(dim=1).mean()
        u_sub = u_t[:, :, 1].sum(dim=1).mean()

        return loss.mean(), u_tot, u_ins, u_del, u_sub, term2.mean()

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

        x_pad_mask = (x_t == self.pad_token)
        # x_ts = [x_t.clone()]

        with tqdm(desc="Euler Sampling") as pbar:
            # while t.max() <= 1 - default_h:
            while t.max() <= 1:
                u_t, sub_logits = self.backbone.forward(x_t, t, x_pad_mask)

                mask_expanded = context_mask.unsqueeze(-1).bool()
                u_t = u_t.masked_fill(mask_expanded, 0.0)
                # Force padded logits to -Inf (so softmax=0)
                sub_logits = sub_logits.masked_fill(mask_expanded, -1e9)

                sub_probs = F.softmax(sub_logits, dim=-1)

                lambda_ins = u_t[:, :, 0]  # Insertion rate        (n_samples, x_seq_len)
                lambda_sub = u_t[:, :, 1]  # Substitution rate     (n_samples, x_seq_len)
                lambda_del = u_t[:, :, 2]  # Deletion rate         (n_samples, x_seq_len)

                # zero out rates for context tokens or pad tokens
                valid_token_mask = (~x_pad_mask & ~context_mask).float()

                lambda_ins = lambda_ins * valid_token_mask
                lambda_sub = lambda_sub * valid_token_mask
                lambda_del = lambda_del * valid_token_mask

                adapt_h = default_h

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
                sub_tokens = torch.full(sub_probs.shape[:2], self.pad_token, dtype=torch.long, device=self.device)

                non_pad_mask = ~x_pad_mask

                if non_pad_mask.any():
                    sub_sampled = torch.multinomial(sub_probs[non_pad_mask], num_samples=1, replacement=True).squeeze(
                        -1)
                    sub_tokens[non_pad_mask] = sub_sampled

                # Apply operations based on masks
                x_t[sub_mask] = sub_tokens[sub_mask]

                x_t = poisson_apply_ins_del_operations(
                    x_t,
                    ins_vals,
                    del_mask,
                    max_seq_len=self.config.model.length,
                    pad_token=self.pad_token,
                    mask_token=self.mask_token
                )
                x_pad_mask = (x_t == self.pad_token)  # Update padding mask after operations

                t = t + adapt_h
                # x_ts.append(x_t.clone())

                # ensure that context is constant
                x_t = torch.where(context_mask, x_0, x_t)

                pbar.update(1)

        return x_t

    def restore_model_and_sample(self, n_steps, eps=1e-5):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        self.backbone.eval()
        samples = self._sample(n_steps=n_steps, eps=eps)
        self.backbone.train()
        return samples


def poisson_apply_ins_del_operations(
        x_t: torch.Tensor,
        ins_vals: torch.Tensor,
        del_mask: torch.Tensor,
        pad_token,
        max_seq_len,
        mask_token
) -> torch.Tensor:
    """
    Apply insertion and deletion operations to a sequence x_t based on the provided masks.
    """
    batch_size, seq_len = x_t.shape
    device = x_t.device

    # Handle simultaneous ins+del as substituting a mask
    replace_mask = (ins_vals > 0) & del_mask
    x_t_modified = x_t.clone()
    x_t_modified[replace_mask] = mask_token
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
    new_pos = pos_idx + cum_ins_before - cum_del  # new pos of tokens shifted by ins/del

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

            flat_indices = torch.arange(total_insertions, device=device)

            # Find start index of each group in the flat array
            group_starts = torch.cat([
                torch.tensor([0], device=device),
                num_insertions_at_loc.cumsum(dim=0)[:-1]
            ])

            # Map every element to its group's start index
            repeated_starts = group_starts.repeat_interleave(num_insertions_at_loc)

            # Subtract start from current to get 1-based offset
            offsets = flat_indices - repeated_starts + 1

            # 5. Calculate final insertion positions
            final_ins_pos = repeated_base_pos + offsets
            # -----------------------------------

            valid_mask = (final_ins_pos >= 0) & (final_ins_pos < max_new_len)
            x_new[repeated_batch_indices[valid_mask], final_ins_pos[valid_mask]] = mask_token
    if max_new_len > max_seq_len:
        print(f"Warning: max_new_len {max_new_len} exceeds max_seq_len {max_seq_len}, truncating.")
        max_new_len = max_seq_len

    return x_new[:, :max_new_len]
