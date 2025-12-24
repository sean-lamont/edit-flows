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
from edit_flow import EditFlowBase, stable_sigmoid_sum
from flow_utils import rm_gap_tokens
from flows import CubicScheduler, x2prob, sample_p


class EditFlowBaseline(EditFlowBase):
    def __init__(
            self,
            config,
            tokenizer: transformers.PreTrainedTokenizer):
        super().__init__(config, tokenizer)

        self.backbone = models.dit_edit_flow_baseline.DITEditFlow(
            self.config, vocab_size=self.V)

        self.time_conditioning = self.config.time_conditioning

        # linear for unmasking (matches up with log linear sigma  = linear alpha with time from t = 1 to t = 0)
        self.default_scheduler = CubicScheduler(a=1.0, b=1.0)

    def get_sched_coeff(self, t, eps=1e-9):
        return (self.default_scheduler.derivative(t) / (1 - self.default_scheduler(t) + eps))

    def _compute_loss(self, batch):
        x_0, x_1, z_0, z_1, t = batch

        z_t = self.sample_zt_sparse(z_0, z_1, t)

        x_t, x_pad_mask, z_gap_mask, z_pad_mask = self.rm_gap_tokens(z_t)

        u_t, sub_logits, ins_logits = self.backbone.forward(x_t, t, x_pad_mask)

        # u_tot = u_t.sum(dim=(1, 2))
        # eps = 1e-9

        sub_rates = u_t[:, :, 0]
        ins_rates = u_t[:, :, 1]
        del_rates = u_t[:, :, 2]

        mask_expanded = x_pad_mask.unsqueeze(-1).bool()

        # Force padded/context logits to -Inf (so softmax=0)
        sub_logits = sub_logits.masked_fill(mask_expanded, -1e9)
        ins_logits = ins_logits.masked_fill(mask_expanded, -1e9)

        eps = 1e-9
        sched_coeff = self.get_sched_coeff(t)

        if self.time_dependent:
            sub_rates = F.softplus(torch.clamp(sub_rates, max=1e6))
            ins_rates = F.softplus(torch.clamp(ins_rates, max=1e6))
            del_rates = F.softplus(torch.clamp(del_rates, max=1e6))

            sub_rates = sub_rates.masked_fill(mask_expanded, 0.0)
            ins_rates = ins_rates.masked_fill(mask_expanded, 0.0)
            del_rates = del_rates.masked_fill(mask_expanded, 0.0)

            u_tot = u_t.sum(dim=(1, 2))

            u_t[:, :, 0] = sub_rates
            u_t[:, :, 1] = ins_rates
            u_t[:, :, 2] = del_rates

            u_t = torch.log(u_t + eps)

        else:  # model outputs time independent logits for ins/sub/del
            sub_rates = sub_rates.masked_fill(mask_expanded, -1e9)
            ins_rates = ins_rates.masked_fill(mask_expanded, -1e9)
            del_rates = del_rates.masked_fill(mask_expanded, -1e9)

            u_t[:, :, 0] = sub_rates
            u_t[:, :, 1] = ins_rates
            u_t[:, :, 2] = del_rates

            if self.rate_scaling:
                # u_tot = stable_sigmoid_sum(sched_coeff, ins_rates) + stable_sigmoid_sum(sched_coeff, sub_rates) + stable_sigmoid_sum(sched_coeff, del_rates)
                u_tot = stable_sigmoid_sum(sched_coeff, u_t, dim=1).sum(dim=-1)
            else:
                # u_tot = ins_rates.sum(dim=-1) + torch.sigmoid(sub_rates).sum(dim=-1) + torch.sigmoid(del_rates).sum(
                #     dim=-1)
                u_tot = torch.sigmoid(u_t).sum(dim=-1)


            u_t = F.logsigmoid(u_t)


        uz_mask = self.make_uz_mask(z_t, z_1)

        sub_mask = uz_mask[:, :, 0]
        ins_mask = uz_mask[:, :, 1]
        del_mask = uz_mask[:, :, 2]

        lse_sub_x = sub_logits.logsumexp(dim=-1, keepdim=True)  # [B, Sx, 1]
        lse_ins_x = ins_logits.logsumexp(dim=-1, keepdim=True)  # [B, Sx, 1]

        packed_features_x = torch.cat([u_t, lse_sub_x, lse_ins_x], dim=-1)

        packed_features_z = self.fill_gap_tokens_with_repeats(
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

        term_ins = (log_rate_ins + target_ins_logits - lse_ins_z) * ins_mask
        term_del = log_rate_del * del_mask
        term_sub = (log_rate_sub + target_sub_logits - lse_sub_z) * sub_mask

        selected_log_ll = term_ins + term_del + term_sub

        if self.rate_scaling:
            term2 = (selected_log_ll * sched_coeff).sum(dim=1)
        else:
            term2 = selected_log_ll.sum(dim=1)

        loss = u_tot - term2

        u_t = torch.exp(u_t)
        u_ins = u_t[:, :, 0].sum(dim=1).mean()
        u_del = u_t[:, :, 2].sum(dim=1).mean()
        u_sub = u_t[:, :, 1].sum(dim=1).mean()

        return loss.mean(), u_tot, u_ins, u_del, u_sub, term2.mean()

    def sample_zt_sparse(self, z_0, z_1, t):
        """
        Samples z_t for Standard Edit Flows.
        1. Interpolates directly between z_0 and z_1 using the default scheduler.
        2. Injects 15% random noise into valid tokens.
           (Falls back to original token if random generation hits Pad/Gap).
        """
        t = t.reshape(-1, 1)
        probs_z1 = self.default_scheduler(t)

        B, L = z_0.shape
        probs_z1 = probs_z1.expand(B, L)

        use_z1 = torch.rand(B, L, device=self.device) < probs_z1
        z_t = torch.where(use_z1, z_1, z_0)

        is_valid_token = (z_t != self.gap_token) & (z_t != self.pad_token)

        noise_prob = 0.15
        noise_mask = (torch.rand_like(z_t, dtype=torch.float) < noise_prob) & is_valid_token

        if noise_mask.any():
            random_tokens = torch.randint(0, self.V, z_t.shape, device=self.device)

            invalid_random = (random_tokens == self.pad_token) | (random_tokens == self.gap_token)
            random_tokens = torch.where(invalid_random, z_t, random_tokens)

            z_t = torch.where(noise_mask, random_tokens, z_t)

        return z_t

    def training_step(self, batch, batch_idx):
        loss, u_tot, u_ins, u_del, u_sub, term2 = self._compute_loss(batch)
        self.log_dict(
            {
                "train_loss": loss,
                "train_u_tot": u_tot.mean(),
                "train_u_ins": u_ins,
                "train_u_del": u_del,
                "train_u_sub": u_sub,
                "train_term2": term2,
            }, prog_bar=True, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.backbone.eval()

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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.backbone.parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1,
                   self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay)

        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler, optimizer=optimizer)
        scheduler_dict = {
            'scheduler': scheduler,
            'interval': 'step',
            'monitor': 'val/loss',
            'name': 'trainer/lr',
        }
        return [optimizer], [scheduler_dict]

    @torch.no_grad()
    def _sample(self, n_steps=None, eps=1e-5):
        """Generate samples from the model."""
        batch_size_per_gpu = self.config.loader.eval_batch_size

        # Lightning auto-casting is not working in this method for some reason
        if n_steps is None:
            n_steps = self.config.sampling.steps

        default_h = 1 / n_steps

        t_min = 0.01

        t = t_min * torch.ones(batch_size_per_gpu, 1, device=self.device)

        x_0 = torch.full((batch_size_per_gpu, 1), 101,
                         device=self.device).long()  # todo parameterise, currently BOS token

        # x_0 = torch.empty((batch_size_per_gpu, 0),
        #                   device=self.device).long()  # todo sample from coupling optionally given x1

        x_t = x_0.clone().to(self.device)

        x_pad_mask = (x_t == self.pad_token)  # Create padding mask for x_t
        # x_ts = [x_t.clone()]

        with tqdm(desc="Euler Sampling") as pbar:
            while t.max() <= 1:
                u_t, sub_logits, ins_logits = self.backbone.forward(x_t, t, x_pad_mask)

                sub_probs = F.softmax(sub_logits, dim=-1)
                ins_probs = F.softmax(ins_logits, dim=-1)

                valid_token_mask = (~x_pad_mask).float()

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
                # x_ts.append(x_t.clone())
                pbar.update(1)

        return x_t

    def restore_model_and_sample(self, n_steps, eps=1e-5):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        self.backbone.eval()
        samples = self._sample(n_steps=n_steps, eps=eps)
        self.backbone.train()
        return samples

    @torch.no_grad()
    def eval_retokenize(self, text_samples, max_length):
        """Retokenizes samples for the eval model.

        Args:
            text_samples: List of sentences generated by the model.
        Returns:
            samples: Samples re-tokenized for the eval model
            attn_mask: Attention mask for the eval model
            eval_context_size: Size of the context for the eval model
        """
        if 'llama2' in self.gen_ppl_eval_model_name_or_path:
            tokenizer_kwargs = {
                'text_samples': text_samples,
                'return_tensors': 'pt',
                'return_token_type_ids': False,
                'return_attention_mask': True,
                'truncation': True,
                'padding': True,
                'max_length': max_length,
            }
            eval_context_size = 4096
        else:
            tokenizer_kwargs = {
                'return_tensors': 'pt',
                'return_token_type_ids': False,
                'return_attention_mask': True,
                'truncation': True,
                'padding': True,
                'max_length': max_length,
            }
            eval_context_size = 1024
        samples = self.eval_model_tokenizer(
            text_samples, **tokenizer_kwargs)
        attn_mask = samples['attention_mask']
        samples = samples['input_ids']
        if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
            attn_mask = attn_mask.to(self.device)
            samples = samples.to(self.device)
        return samples, attn_mask, eval_context_size

    @torch.no_grad()
    def compute_generative_perplexity(
            self,
            text_samples: typing.List[str],
            retokenize: bool = True,
            max_length: typing.Optional[int] = None) -> None:
        """Compute the generative perplexity of the model.

        Args:
            text_samples: List of sentences generated by the model.

        Returns:
            Perplexity of the generated text under a different
            pre-trained AR model (e.g., GPT2).
        """
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        eval_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.gen_ppl_eval_model_name_or_path).eval()
        if max_length is None:
            max_length = self.config.model.length
        if 'llama2' not in self.gen_ppl_eval_model_name_or_path:
            eval_model = eval_model.to(self.device)
        # Re-tokenize using eval model's tokenizer
        if retokenize:
            (samples, attn_mask,
             eval_context_size) = self.eval_retokenize(
                text_samples, max_length=max_length)
        else:
            samples = text_samples
            attn_mask = torch.ones(samples.shape).to(self.device)
            eval_context_size = samples.shape[-1]
        batch_size = min(
            self.config.eval.perplexity_batch_size,
            samples.shape[0])
        num_batches = samples.shape[0] // batch_size
        for i in range(num_batches):
            _samples = torch.split(
                samples[i * batch_size: (i + 1) * batch_size],
                eval_context_size,
                dim=-1)
            _attn_mask = torch.split(
                attn_mask[i * batch_size: (i + 1) * batch_size],
                eval_context_size,
                dim=-1)
            for (sample_chunk, attn_mask_chunk) in zip(
                    _samples, _attn_mask):
                logits = eval_model(
                    sample_chunk, attention_mask=attn_mask_chunk)[0]
                logits = logits.transpose(-1, -2)

                nlls = F.cross_entropy(logits[..., :-1],
                                       sample_chunk[..., 1:],
                                       reduction='none')
                first_eos = (sample_chunk == self.eval_model_tokenizer \
                             .eos_token_id).cumsum(-1) == 1
                token_mask = (
                        sample_chunk
                        != self.eval_model_tokenizer.eos_token_id)
                self.gen_ppl_metric.update(
                    nlls, first_eos[..., 1:] + token_mask[..., 1:])


def apply_ins_del_operations(
        x_t: torch.Tensor,
        ins_mask: torch.Tensor,
        del_mask: torch.Tensor,
        ins_tokens,
        max_seq_len,
        pad_token,
) -> torch.Tensor:
    """
    Apply insertion and deletion operations to a sequence x_t based on the provided masks.
    """
    batch_size, seq_len = x_t.shape
    device = x_t.device

    # Handle simultaneous ins+del as substitutions
    replace_mask = ins_mask & del_mask
    x_t_modified = x_t.clone()
    x_t_modified[replace_mask] = ins_tokens[replace_mask]

    # Update ins/del masks after handling replacements
    eff_ins_mask = ins_mask & ~replace_mask
    eff_del_mask = del_mask & ~replace_mask

    # Compute new lengths after applying ins/del operations
    xt_pad_mask = (x_t == pad_token)  # (batch_size, seq_len)
    xt_seq_lens = (~xt_pad_mask).sum(dim=1)  # (batch_size,)
    new_lengths = xt_seq_lens + eff_ins_mask.sum(dim=1) - eff_del_mask.sum(dim=1)
    max_new_len = int(new_lengths.max().item())

    if max_new_len <= 0:
        print(f"Unexpected max_new_len <= 0: {max_new_len}, did we delete everything?")
        return torch.full((batch_size, 1), pad_token, dtype=x_t.dtype, device=device)

    # Pre-allocate result
    x_new = torch.full((batch_size, max_new_len), pad_token, dtype=x_t.dtype, device=device)

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
