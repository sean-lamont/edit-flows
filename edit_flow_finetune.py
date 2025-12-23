import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from tqdm import tqdm

from edit_flow import EditFlow, stable_sigmoid_sum
from flow_utils import rm_gap_tokens


class LLaDABackbone(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config

        # Load LLaDA
        print(f"Loading LLaDA Backbone: {config.model_name}")
        self.base_model = transformers.AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        self.hidden_size = self.base_model.config.hidden_size
        self.vocab_size = vocab_size

        self.ins_head = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.GELU(),
            nn.Linear(1024, 1)  # Output raw log_k
        )

        self.del_head = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.GELU(),
            nn.Linear(1024, 1)  # Output raw logits
        )

        self.sub_head = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.GELU(),
            nn.Linear(1024, 1)  # Output raw logits
        )

        if hasattr(self.base_model, 'lm_head'):
            self.content_head = self.base_model.lm_head
        else:
            self.content_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)

    def forward(self, x, t=None, attention_mask=None):
        outputs = self.base_model(x, attention_mask=attention_mask, output_hidden_states=True)
        h = outputs.last_hidden_state

        ins_pred = self.ins_head(h)
        sub_pred = self.sub_head(h)
        del_pred = self.del_head(h)

        sub_logits = self.content_head(h)

        rates = torch.cat([sub_pred, ins_pred, del_pred], dim=-1)

        return rates, sub_logits


class EditFlowFineTune(EditFlow):
    def __init__(
            self,
            config,
            tokenizer: transformers.PreTrainedTokenizer):
        super().__init__(config, tokenizer)

        self.backbone = LLaDABackbone(
            self.config, vocab_size=self.V)

        self.gap_token = 126084  # for LLADA , reserved_token0

    def _compute_loss(self, batch):
        # context mask = (bsz, seq_len) where we keep original context (i.e. x0, z0, z1, x1 should all have context_ids the same)
        x_0, x_1, z_0, z_1, t, context_mask = batch

        z_t = self.sample_zt_sparse(z_0, z_1, t)

        x_t, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z_t, pad_token=self.pad_token, gap_token=self.gap_token)

        # we set x_t to x_1 for context_ids (assumes x0, x1, z0, z1 all have context at same ids!)

        x_t = torch.where(context_mask, x_1, x_t)

        u_t, sub_logits = self.backbone.forward(x_t, t, x_pad_mask)

        sub_rates = u_t[:, :, 0]
        ins_rates = u_t[:, :, 1]
        del_rates = u_t[:, :, 2]

        mask_expanded = (context_mask | x_pad_mask).unsqueeze(-1).bool()

        # Force padded/context logits to -Inf (so softmax=0)
        sub_logits = sub_logits.masked_fill(mask_expanded, -1e9)

        # rates always positive for insert (time dependent = full rate, time independent = predicted # inserts)
        ins_rates = F.softplus(torch.clamp(ins_rates, max=1e6))
        ins_rates = ins_rates.masked_fill(mask_expanded, 0.0)

        eps = 1e-9
        sched_coeff = self.get_sched_coeff(t, z_0, z_1, z_t)

        if self.time_dependent:  # model outputs the full rate prediction
            sub_rates = F.softplus(torch.clamp(sub_rates, max=1e6))
            sub_rates = sub_rates.masked_fill(mask_expanded, 0.0)

            del_rates = F.softplus(torch.clamp(del_rates, max=1e6))
            del_rates = del_rates.masked_fill(mask_expanded, 0.0)

            u_tot = u_t.sum(dim=(1, 2))

            u_t[:, :, 0] = sub_rates
            u_t[:, :, 1] = ins_rates
            u_t[:, :, 2] = del_rates

            u_t = torch.log(u_t + eps)

        else:  # model outputs time independent logit for a sub/delete
            sub_rates = sub_rates.masked_fill(mask_expanded, -1e9)
            del_rates = del_rates.masked_fill(mask_expanded, -1e9)

            if self.rate_scaling:
                u_tot = (ins_rates * sched_coeff).sum(dim=-1) + stable_sigmoid_sum(sched_coeff,
                                                                                   sub_rates) + stable_sigmoid_sum(
                    sched_coeff, del_rates)
            else:
                u_tot = ins_rates.sum(dim=-1) + torch.sigmoid(sub_rates).sum(dim=-1) + torch.sigmoid(del_rates).sum(
                    dim=-1)

            u_t[:, :, 0] = F.logsigmoid(sub_rates)
            u_t[:, :, 1] = torch.log(ins_rates + eps)
            u_t[:, :, 2] = F.logsigmoid(del_rates)

            u_t = u_t

        uz_mask = self.make_uz_mask(z_t, z_1)

        target_sub = uz_mask[:, :, 0] & ~context_mask
        target_ins = uz_mask[:, :, 1] & ~context_mask
        target_del = uz_mask[:, :, 2] & ~context_mask

        log_sum_exp_x = sub_logits.logsumexp(dim=-1)

        log_sum_exp_z = self.fill_gap_tokens_with_repeats(
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

        uz_log_rates = self.fill_gap_tokens_with_repeats(u_t, z_gap_mask, z_pad_mask)

        log_rate_ins = uz_log_rates[:, :, 0]
        log_rate_sub = uz_log_rates[:, :, 1]
        log_rate_del = uz_log_rates[:, :, 2]

        selected_log_ll = (
                (log_rate_ins * target_ins) +
                (log_rate_del * target_del) +
                ((log_rate_sub - vocab_nll) * target_sub)
        )

        if self.rate_scaling:
            term2 = (selected_log_ll * sched_coeff).sum(dim=1)
        else:
            term2 = selected_log_ll.sum(dim=1)

        loss = u_tot - term2

        u_ins = torch.exp(u_t[:, :, 0]).sum(dim=1).mean()
        u_del = torch.exp(u_t[:, :, 2]).sum(dim=1).mean()
        u_sub = torch.exp(u_t[:, :, 1]).sum(dim=1).mean()

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

                x_t = self.apply_ins_del_ops(
                    x_t,
                    ins_vals,
                    del_mask,
                )
                x_pad_mask = (x_t == self.pad_token)  # Update padding mask after operations

                t = t + adapt_h
                # x_ts.append(x_t.clone())

                # ensure that context is constant
                x_t = torch.where(context_mask, x_0, x_t)

                pbar.update(1)

        return x_t
