import datetime
import html
import math
import os
import typing

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
from flows import CubicScheduler

LOG2 = math.log(2)


class NLL(torchmetrics.aggregation.MeanMetric):
    pass


def stable_sigmoid_sum(a, b, dim=-1):
    """
    Computes sum(a * sigmoid(b)) stably, assuming a > 0.
    """
    log_a = torch.log(a + 1e-45)

    # log(a * sig(b)) = log(a) + log_sigmoid(b)
    log_terms = log_a + F.logsigmoid(b)

    sum_in_log_space = torch.logsumexp(log_terms, dim=dim)

    return torch.exp(sum_in_log_space)


class BPD(NLL):
    def compute(self) -> Tensor:
        """Computes the bits per dimension.

        Returns:
          bpd
        """
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> Tensor:
        """Computes the Perplexity.

        Returns:
         Perplexity
        """
        return torch.exp(self.mean_value / self.weight)


class EditFlowBase(L.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.tokenizer = tokenizer
        # whether the model outputs the rate, or reparameterised time independant score
        self.time_dependent = self.config.get('time_dependent', True)

        # whether to include the scheduler term in the loss calculation
        self.rate_scaling = self.config.get('rate_scaling', True)

        self.gen_ppl_eval_model_name_or_path = self.config.eval. \
            gen_ppl_eval_model_name_or_path

        self.eval_model_tokenizer = transformers.AutoTokenizer. \
            from_pretrained(self.gen_ppl_eval_model_name_or_path)

        self.eval_model_tokenizer = transformers.AutoTokenizer. \
            from_pretrained(self.gen_ppl_eval_model_name_or_path)

        if self.eval_model_tokenizer.pad_token is None:
            self.eval_model_tokenizer.pad_token = \
                self.eval_model_tokenizer.eos_token
            self.eval_model_tokenizer.pad_token_id = \
                self.eval_model_tokenizer.eos_token_id

        self.pad_token = self.tokenizer.pad_token_id

        self.gap_token = self.config.get('gap_token', 3)  # todo hardcoded for now, use unused token or manually add one

        self.fast_forward_epochs = None
        self.fast_forward_batches = None

        # generative perplexity
        self.gen_ppl_metric = Perplexity()

        self.lr = self.config.optim.lr

        self.V = self.tokenizer.vocab_size

    def make_uz_mask(self,
                     z_t: torch.Tensor,
                     z_1: torch.Tensor,
                     ) -> torch.Tensor:
        """
        Create a mask for u_cat for indexing the output rate tensor based on differences between z_t and z_1.
        For each position i where z_t and z_1 differ, we index as follows:

        - z_t[i] = GAP_TOKEN & z_1[i] = c => u_mask[i, insert] = 1
        - z_t[i] = c & z_1[i] = GAP_TOKEN => u_mask[i, delete] = 1
        - z_t[i] = c1 & z_1[i] = c2 => u_mask[i, substitute] = 1
        """
        batch_size, z_seq_len = z_t.shape
        # n_ops = vocab_size + 2  # substitute + delete + insert

        z_neq = (z_t != z_1) & (z_t != self.pad_token) & (z_1 != self.pad_token)
        z_ins = (z_t == self.gap_token) & (z_1 != self.gap_token) & z_neq  # (batch_size, z_seq_len)
        z_del = (z_t != self.gap_token) & (z_1 == self.gap_token) & z_neq  # (batch_size, z_seq_len)
        z_sub = z_neq & ~z_ins & ~z_del  # (batch_size, z_seq_len)

        # mask (batch_size, z_seq_len, u_ops) where 1 indicates operation that bring z_t closer to z_1
        u_mask = torch.zeros((batch_size, z_seq_len, 3), dtype=torch.bool, device=z_t.device)

        u_mask[:, :, 0][z_sub] = True
        u_mask[:, :, 1][z_ins] = True
        u_mask[:, :, 2][z_del] = True

        assert z_neq.sum() == (z_ins | z_del | z_sub).sum(), "Mismatch in number of edits"
        assert z_neq.sum() == u_mask.sum(), "Mismatch in number of edits in mask"

        return u_mask

    def on_load_checkpoint(self, checkpoint):
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        self.fast_forward_epochs = checkpoint['loops'][
            'fit_loop']['epoch_progress']['current']['completed']
        self.fast_forward_batches = checkpoint['loops'][
            'fit_loop']['epoch_loop.batch_progress'][
            'current']['completed']

    def on_save_checkpoint(self, checkpoint):
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
        # behind, so we're using the optimizer's progress.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['total'][
            'completed'] = checkpoint['loops']['fit_loop'][
                               'epoch_loop.automatic_optimization.optim_progress'][
                               'optimizer']['step']['total'][
                               'completed'] * self.trainer.accumulate_grad_batches
        checkpoint['loops']['fit_loop'][
            'epoch_loop.batch_progress']['current'][
            'completed'] = checkpoint['loops']['fit_loop'][
                               'epoch_loop.automatic_optimization.optim_progress'][
                               'optimizer']['step']['current'][
                               'completed'] * self.trainer.accumulate_grad_batches
        # _batches_that_stepped tracks the number of global steps, not the number
        # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
        checkpoint['loops']['fit_loop'][
            'epoch_loop.state_dict'][
            '_batches_that_stepped'] = checkpoint['loops']['fit_loop'][
            'epoch_loop.automatic_optimization.optim_progress'][
            'optimizer']['step']['total']['completed']

    def on_train_start(self):
        # Adapted from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
        distributed = (
                self.trainer._accelerator_connector.use_distributed_sampler
                and self.trainer._accelerator_connector.is_distributed)
        if distributed:
            sampler_cls = dataloader.FaultTolerantDistributedSampler
        else:
            sampler_cls = dataloader.RandomFaultTolerantSampler
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            if hasattr(dl.sampler, 'shuffle'):
                dl_sampler = sampler_cls(
                    dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)
            if (distributed
                    and self.fast_forward_epochs is not None
                    and self.fast_forward_batches is not None):
                print('Loading Sampler Checkpoint...')
                dl_sampler.load_state_dict({
                    'epoch': self.fast_forward_epochs,
                    'counter': (self.fast_forward_batches
                                * self.config.loader.batch_size)})

            updated_dls.append(
                torch.utils.data.DataLoader(
                    dl.dataset,
                    batch_size=self.config.loader.batch_size,
                    num_workers=self.config.loader.num_workers,
                    pin_memory=self.config.loader.pin_memory,
                    sampler=dl_sampler,
                    shuffle=False,
                    persistent_workers=True,
                    collate_fn=dl.collate_fn
                ))

        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def forward(self, x, sigma):  # sigma is just t for our case
        with torch.cuda.amp.autocast(dtype=torch.float32):
            logits = self.backbone(x, sigma)

        return logits

    def on_train_epoch_start(self):
        self.backbone.train()

    def restore_model_and_sample(self, n_steps, eps=1e-5):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        self.backbone.eval()
        samples = self._sample(n_steps=n_steps, eps=eps)
        self.backbone.train()
        return samples

    def on_validation_epoch_start(self):
        self.backbone.eval()

    def fill_gap_tokens_with_repeats(self,
                                     x_ut: torch.Tensor,
                                     z_gap_mask: torch.Tensor,
                                     z_pad_mask: torch.Tensor,
                                     ):
        batch_size, _ = z_gap_mask.shape
        _, x_seq_len, _ = x_ut.shape

        # Use cumsum on non-gap positions to point to the last valid non-gap position
        non_gap_mask = ~z_gap_mask  # Invert mask to get non-gap positions
        indices = non_gap_mask.cumsum(dim=1) - 1  # (batch_size, z_seq_len)
        indices = indices.clamp(min=0, max=x_seq_len - 1)  # Ensure indices are within bounds

        # Use indices to gather from x_ut
        batch_indices = torch.arange(batch_size, device=x_ut.device).unsqueeze(1)
        result = x_ut[batch_indices, indices]  # (batch_size, z_seq_len, vocab_size) (indexing with [b, 1], [b, z_len])
        result[z_pad_mask] = 0  # Set pad positions to 0
        return result

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

    def sample_and_log(self, n_steps):
        all_gen_lens = []
        # samples, text_samples = None, None
        samples = []
        for _ in range(self.config.sampling.num_sample_batches):
            samples_ = self._sample(n_steps=n_steps)

            # Calculate non-padded lengths
            batch_lens = (samples_ != self.tokenizer.pad_token_id).sum(dim=1).float()
            all_gen_lens.append(batch_lens)

            sample = [s[s != self.tokenizer.pad_token_id] for s in samples_]
            samples.extend([s[s != self.tokenizer.bos_token_id] for s in
                            sample])  # = input_ids[input_ids != tokenizer.pad_token_id]

        # Decode the samples to be re-tokenized by eval model
        text_samples = self.tokenizer.batch_decode(samples)

        if self.config.eval.compute_generative_perplexity:
            self.gen_ppl_metric.reset()
            self.compute_generative_perplexity(text_samples)

        if all_gen_lens:
            all_gen_lens = torch.cat(all_gen_lens)
            self.log_dict({
                f"val/gen_len_mean_{n_steps}": all_gen_lens.mean(),
                f"val/gen_len_std_{n_steps}": all_gen_lens.std(),
            }, prog_bar=True, on_epoch=True, sync_dist=True)

        if self.trainer.global_rank == 0 and hasattr(self.trainer.logger, 'log_table'):
            # Log the last generated samples
            text_samples = text_samples[:self.config.sampling.num_sample_log]

            for sample in text_samples[:min(self.config.sampling.num_sample_log, 2)]:
                print('Sample: ')
                print(sample)
                print('\n')

            self.trainer.logger.log_table(
                key=f'samples_{n_steps}_steps@global_step{self.global_step}',
                columns=['Generated Samples'],
                data=[[s] for s in text_samples])

        if self.config.eval.compute_generative_perplexity:
            # print (f'{self.gen_ppl_metric.compute()}')
            self.log(f'val/gen_ppl_{n_steps}',
                     self.gen_ppl_metric.compute(),
                     on_epoch=True,
                     on_step=False,
                     sync_dist=True,
                     prog_bar=True)

    def on_validation_epoch_end(self):
        # Keeps original logging
        if ((self.config.eval.compute_perplexity_on_sanity
             or not self.trainer.sanity_checking)
                and self.config.eval.generate_samples):

            # steps = [1, 2, 4, 8, 16, 32, 64, 128, 1024]
            steps = [2, 4, 8, 16, 32, 64, 128]
            for step in steps:
                self.sample_and_log(step)

    def rm_gap_tokens(self, z: torch.Tensor):
        """
        Remove gap tokens from a batched tensor and right-pad with PAD_TOKEN.
        """
        batch_size, z_len = z.shape
        device = z.device

        z_gap_mask = (z == self.gap_token)
        z_pad_mask = (z == self.pad_token)

        # Mask for tokens to keep (neither GAP nor PAD)
        keep_mask = ~z_gap_mask & ~z_pad_mask

        # Get the values and their original batch indices
        kept_values = z[keep_mask]
        batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, z_len)[keep_mask]

        # Calculate new positions and scatter into a new tensor
        new_lengths = keep_mask.sum(dim=1)
        max_len = new_lengths.max().item()

        x = torch.full((batch_size, max_len), self.pad_token, dtype=z.dtype, device=device)

        new_pos = torch.arange(z_len, device=device).unsqueeze(0).expand(batch_size, -1)[keep_mask] - \
                  z_gap_mask.cumsum(dim=1)[keep_mask]

        x[batch_indices, new_pos] = kept_values

        x_pad_mask = (x == self.pad_token)
        assert ((~x_pad_mask).sum(1) + z_gap_mask.sum(1)).equal((~z_pad_mask).sum(1))
        return x, x_pad_mask, z_gap_mask, z_pad_mask


# Unconditional Edit Flow model
class EditFlow(EditFlowBase):
    def __init__(
            self,
            config,
            tokenizer: transformers.PreTrainedTokenizer):
        super().__init__(config, tokenizer)

        if (not hasattr(self.tokenizer, 'mask_token')
                or self.tokenizer.mask_token is None):
            self.mask_token = self.V
            self.V += 1
        else:
            self.mask_token = self.tokenizer.mask_token_id

        self.backbone = models.dit_edit_flow.DITEditFlow(
            self.config, vocab_size=self.V)

        self.time_conditioning = self.config.time_conditioning

        # higher mass earlier for structure prediction, giving more time for unmasking
        self.mask_scheduler = CubicScheduler(a=3.0, b=0.0)

        # linear for unmasking (matches up with log linear sigma  = linear alpha with time from t = 1 to t = 0)
        self.default_scheduler = CubicScheduler(a=1.0, b=1.0)

    def get_sched_coeff(self, t, z_0, z_1, z_t, eps=1e-9):
        default_coeff = (self.default_scheduler.derivative(t) / (1 - self.default_scheduler(t) + eps))
        ins_coeff = (self.mask_scheduler.derivative(t) / (1 - self.mask_scheduler(t) + eps))

        mask_sub_coeff = (self.mask_scheduler.derivative(t) * self.default_scheduler(t)
                          + self.mask_scheduler(t) * self.default_scheduler.derivative(t)) / (
                                 self.mask_scheduler(t) * (1 - self.default_scheduler(t)) + eps)

        z_ins_event = (z_0 == self.gap_token) & (z_1 != self.gap_token) & (z_0 != z_1)

        mask_ids = (z_t == self.mask_token) & (z_1 != self.mask_token) & (z_0 != self.mask_token) & (z_1 != z_0)

        sched_coeff = torch.where(z_ins_event, ins_coeff, default_coeff)

        sched_coeff = torch.where(mask_ids, mask_sub_coeff, sched_coeff)

        return sched_coeff

    def _compute_loss(self, batch):
        x_0, x_1, z_0, z_1, t = batch
        bsz = x_0.shape[0]

        z_t = self.sample_zt(z_0, z_1, t)
        x_t, x_pad_mask, z_gap_mask, z_pad_mask = self.rm_gap_tokens(z_t)

        sched_coeff_z = self.get_sched_coeff(t, z_0, z_1, z_t)

        if not self.time_dependent:
            t = torch.ones_like(t).to(self.device)

        u_t_logits, sub_vocab_logits = self.backbone.forward(x_t, t, x_pad_mask)

        raw_sub = u_t_logits[:, :, 0]
        raw_ins = u_t_logits[:, :, 1]
        raw_del = u_t_logits[:, :, 2]

        # Mask padding logits
        mask_expanded = x_pad_mask.unsqueeze(-1)

        sub_vocab_logits = sub_vocab_logits.masked_fill(mask_expanded, -1e9)

        if self.time_dependent:
            r_sub = F.softplus(torch.clamp(raw_sub, max=1e6)).masked_fill(x_pad_mask, 0)
            r_ins = F.softplus(torch.clamp(raw_ins, max=1e6)).masked_fill(x_pad_mask, 0)
            r_del = F.softplus(torch.clamp(raw_del, max=1e6)).masked_fill(x_pad_mask, 0)

            u_tot = (r_ins + r_sub + r_del).sum(dim=-1)

            eps = 1e-9
            log_r_sub = torch.log(r_sub + eps)
            log_r_ins = torch.log(r_ins + eps)
            log_r_del = torch.log(r_del + eps)

        else:  # Time Independent

            r_ins = F.softplus(raw_ins).masked_fill(x_pad_mask, 0)
            r_sub = torch.sigmoid(raw_sub).masked_fill(x_pad_mask, 0)
            r_del = torch.sigmoid(raw_del).masked_fill(x_pad_mask, 0)

            if self.rate_scaling:
                sched_coeff_x = torch.zeros_like(x_t, dtype=u_t_logits.dtype)

                if sched_coeff_z.dim() > 1 and sched_coeff_z.shape[1] == z_t.shape[1]:
                    mask_z = ~z_gap_mask
                    ranks = mask_z.cumsum(dim=1) - 1
                    valid_z = mask_z & (ranks < x_t.shape[1])

                    values = sched_coeff_z[valid_z]
                    dest_cols = ranks[valid_z]
                    dest_rows = torch.arange(bsz, device=x_t.device).unsqueeze(1).expand_as(z_t)[valid_z]

                    values = values.to(dtype=sched_coeff_x.dtype)
                    sched_coeff_x[dest_rows, dest_cols] = values

                else:
                    sched_coeff_x = sched_coeff_z

                u_tot = (r_ins * sched_coeff_x).sum(dim=-1) + \
                        (r_sub * sched_coeff_x).sum(dim=-1) + \
                        (r_del * sched_coeff_x).sum(dim=-1)
            else:
                u_tot = r_ins.sum(dim=-1) + r_sub.sum(dim=-1) + r_del.sum(dim=-1)

            # Log Rates
            log_r_ins = torch.log(r_ins + 1e-9)
            log_r_sub = F.logsigmoid(raw_sub)
            log_r_del = F.logsigmoid(raw_del)

        lse_sub_x = sub_vocab_logits.logsumexp(dim=-1)  # , keepdim=True)

        packed_features_x = torch.stack([log_r_sub, log_r_ins, log_r_del, lse_sub_x], dim=-1)

        packed_features_z = self.fill_gap_tokens_with_repeats(
            packed_features_x, z_gap_mask, z_pad_mask
        )

        log_rate_sub_z = packed_features_z[..., 0]
        log_rate_ins_z = packed_features_z[..., 1]
        log_rate_del_z = packed_features_z[..., 2]
        lse_sub_z = packed_features_z[..., 3]

        uz_mask = self.make_uz_mask(z_t, z_1)
        sub_mask = uz_mask[:, :, 0]
        ins_mask = uz_mask[:, :, 1]
        del_mask = uz_mask[:, :, 2]

        non_gap_mask = ~z_gap_mask
        x_indices = non_gap_mask.cumsum(dim=1) - 1
        x_indices = x_indices.clamp(min=0, max=x_t.shape[1] - 1)

        valid_vocab_limit = sub_vocab_logits.size(-1) - 1
        safe_z1 = z_1.clamp(min=0, max=valid_vocab_limit)
        batch_idx_seq = torch.arange(bsz, device=self.device).unsqueeze(1)

        target_logits_z = sub_vocab_logits[batch_idx_seq, x_indices, safe_z1]

        vocab_nll = lse_sub_z - target_logits_z

        selected_log_ll = (
                (log_rate_ins_z * ins_mask) +
                (log_rate_del_z * del_mask) +
                ((log_rate_sub_z - vocab_nll) * sub_mask)
        )

        if self.rate_scaling:
            term2 = (selected_log_ll * sched_coeff_z).sum(dim=1)
        else:
            term2 = selected_log_ll.sum(dim=1)

        loss = u_tot - term2

        with torch.no_grad():
            u_ins_mean = r_ins.sum(dim=1).mean()
            u_del_mean = r_del.sum(dim=1).mean()
            u_sub_mean = r_sub.sum(dim=1).mean()

        return loss.mean(), u_tot.mean(), u_ins_mean, u_del_mean, u_sub_mean, term2.mean()

    def sample_zt(self, z_0, z_1, t):
        t = t.reshape(-1, 1)  # [Batch, 1]
        mask_t = self.mask_scheduler(t)
        default_t = self.default_scheduler(t)

        w_z0_ins = 1 - mask_t
        w_mask_ins = mask_t * (1 - default_t)
        w_z1_ins = mask_t * default_t

        w_z0_std = 1 - default_t
        w_mask_std = torch.zeros_like(default_t)
        w_z1_std = default_t

        z_neq = (z_0 != z_1) & (z_0 != self.pad_token) & (z_1 != self.pad_token)
        is_ins = (z_0 == self.gap_token) & (z_1 != self.gap_token) & z_neq

        B, L = z_0.shape
        w_z0 = torch.where(is_ins, w_z0_ins, w_z0_std).expand(B, L)
        w_mask = torch.where(is_ins, w_mask_ins, w_mask_std).expand(B, L)
        w_z1 = torch.where(is_ins, w_z1_ins, w_z1_std).expand(B, L)

        probs = torch.stack([w_z0, w_mask, w_z1], dim=-1)
        flat_probs = probs.view(-1, 3)
        choices = torch.multinomial(flat_probs, 1).view(B, L)

        random_tokens = torch.randint(0, self.V, z_0.shape, device=self.device)

        safe_fallback = torch.tensor(self.mask_token, device=self.device)
        invalid_mask = (random_tokens == self.pad_token) | (random_tokens == self.gap_token)
        random_tokens = torch.where(invalid_mask, safe_fallback, random_tokens)

        noise_prob = 0.15
        use_random = torch.rand_like(z_0, dtype=torch.float) < noise_prob

        mask_or_noise = torch.where(use_random,
                                    random_tokens,
                                    torch.tensor(self.mask_token, device=self.device))

        z_t = torch.where(choices == 0, z_0,
                          torch.where(choices == 1, mask_or_noise,
                                      z_1))

        return z_t

    @torch.no_grad()
    def _sample(self, n_steps=None, eps=1e-5, one_shot=False, vis=False,
                viz_path='/home/sean/Documents/edit-flows/sample_vis/samples'):
        """Generate samples from the model."""
        batch_size_per_gpu = self.config.loader.eval_batch_size

        # Lightning auto-casting is not working in this method for some reason
        if n_steps is None:
            n_steps = self.config.sampling.steps

        n_steps = n_steps - 1

        default_h = 1 / n_steps

        t_min = 0.00

        t = t_min * torch.ones(batch_size_per_gpu, 1, device=self.device)

        x_0 = torch.full((batch_size_per_gpu, 1), 101,
                         device=self.device).long()  # todo parameterise, currently BOS token

        x_t = x_0.clone().to(self.device)

        x_pad_mask = (x_t == self.pad_token)  # Create padding mask for x_t
        # x_ts = [x_t.clone()]
        history = []
        with tqdm(desc="Euler Sampling") as pbar:
            # while t.max() <= 1 - default_h:
            # while t.max() <= 1:
            for step_i in range(n_steps + 1):
                # if not self.time_dependent:
                #     u_t, sub_logits = self.backbone.forward(x_t, torch.ones_like(t).to(self.device), x_pad_mask)
                # else:
                u_t, sub_logits = self.backbone.forward(x_t, t, x_pad_mask)

                sub_probs = F.softmax(sub_logits.float(), dim=-1)

                lambda_sub = u_t[:, :, 0].float()
                lambda_ins = u_t[:, :, 1].float()
                lambda_del = u_t[:, :, 2].float()

                # --- CAPTURE VIZ DATA ---

                if not self.time_dependent:  # move logits to probabilities
                    lambda_sub = torch.sigmoid(lambda_sub)
                    lambda_del = torch.sigmoid(lambda_del)
                    lambda_ins = F.softplus(torch.clamp(lambda_ins, max=1e6))
                else:
                    lambda_sub = F.softplus(torch.clamp(lambda_sub, max=1e6))
                    lambda_ins = F.softplus(torch.clamp(lambda_ins, max=1e6))
                    lambda_del = F.softplus(torch.clamp(lambda_del, max=1e6))

                valid_token_mask = (~x_pad_mask)  # .float()

                lambda_ins = torch.where(valid_token_mask, lambda_ins, 0.0)
                lambda_sub = torch.where(valid_token_mask, lambda_sub, 0.0)
                lambda_del = torch.where(valid_token_mask, lambda_del, 0.0)

                if not self.time_dependent and not one_shot:  # scale raw count/bernoulli predictions by sampler rate
                    default_coeff = (self.default_scheduler.derivative(t) / (1 - self.default_scheduler(t) + eps))
                    ins_coeff = (self.mask_scheduler.derivative(t) / (1 - self.mask_scheduler(t) + eps))

                    mask_sub_coeff = (self.mask_scheduler.derivative(t) * self.default_scheduler(t)
                                      + self.mask_scheduler(t) * self.default_scheduler.derivative(t)) / (
                                             self.mask_scheduler(t) * (1 - self.default_scheduler(t)) + eps)

                    mask_ids = (x_t == self.mask_token)

                    sub_coeff = torch.where(mask_ids, mask_sub_coeff, default_coeff)

                    lambda_sub = lambda_sub * sub_coeff
                    lambda_ins = lambda_ins * ins_coeff
                    lambda_del = lambda_del * default_coeff

                if one_shot:  # samples full completion each step (only makes sense for time independent)
                    adapt_h = 1
                else:
                    adapt_h = default_h

                if vis:
                    step_data = {
                        "step": step_i,
                        "t": t[0].item(),
                        "tokens": self.tokenizer.convert_ids_to_tokens(
                            [a for a in x_t[0].cpu().numpy() if a != self.tokenizer.pad_token_id]),
                        "rates": {
                            "ins": lambda_ins[0].float().cpu().numpy(),
                            "del": lambda_del[0].float().cpu().numpy(),
                            "sub": lambda_sub[0].float().cpu().numpy()
                        },
                        "adapt_h": adapt_h,
                        'is_gen': valid_token_mask[0].cpu().numpy()
                    }
                    history.append(step_data)

                ins_vals = torch.poisson(adapt_h * lambda_ins).long()

                del_sub_mask = torch.rand(
                    size=lambda_sub.shape, device=lambda_sub.device
                ) < 1 - torch.exp(-adapt_h * (lambda_sub + lambda_del))

                # For deletion/substitution, sample based on the relative rates
                prob_del = torch.where(
                    del_sub_mask, lambda_del / (lambda_sub + lambda_del), torch.zeros_like(lambda_del))

                del_mask = torch.bernoulli(prob_del).bool()

                sub_mask = del_sub_mask & ~del_mask

                # --- NEW LOGIC: FORCE MASK SUBSTITUTION ---
                if step_i == n_steps:
                    # Identify all current mask tokens
                    mask_token_locs = (x_t == self.mask_token)

                    # Force substitution on these tokens
                    sub_mask = sub_mask | mask_token_locs

                    ins_vals.zero_()

                # assert sub_mask.sum() + del_mask.sum() == del_sub_mask.sum()

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

                t = torch.where(t + adapt_h > 0.99, 0.99, t + adapt_h)
                # x_ts.append(x_t.clone())
                pbar.update(1)

        if vis and len(history) > 0:
            self._export_to_html(history, viz_path + f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.html"
                                 )  # add datetime
            print(f"Visualization saved to {viz_path}")
        return x_t

    def _export_to_html(self, history, output_path):
        html_content = [
            """
            <html>
            <head>
                <style>
                    body { font-family: monospace; background: #f5f5f5; padding: 20px; }
                    .step-container { background: white; margin-bottom: 20px; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                    .step-header { font-weight: bold; margin-bottom: 10px; color: #333; }
                    .sequence { display: flex; flex-wrap: wrap; gap: 4px; }
                    .token-box { 
                        display: inline-flex; flex-direction: column; align-items: center; 
                        border: 1px solid #ddd; padding: 2px; border-radius: 4px; 
                        position: relative; min-width: 20px; text-align: center;
                    }
                    .token-text { font-size: 14px; padding: 2px 4px; z-index: 2; }

                    /* Visual Indicators */
                    .indicator-bar { height: 4px; width: 100%; margin-top: 2px; display: flex; }
                    .rate-ins { background-color: #4CAF50; height: 100%; } /* Green for Insert */

                    .overlay { position: absolute; top: 0; left: 0; right: 0; bottom: 0; opacity: 0.3; pointer-events: none; z-index: 1; }

                    /* Tooltip */
                    .token-box:hover .tooltip { visibility: visible; }
                    .tooltip {
                        visibility: hidden; width: 200px; background-color: #333; color: #fff; text-align: left;
                        border-radius: 6px; padding: 5px; position: absolute; z-index: 10;
                        bottom: 100%; left: 50%; margin-left: -100px; font-size: 11px;
                        white-space: pre-wrap;
                    }
                </style>
            </head>
            <body>
            <h1>Targeted Edit Flow Generation Process</h1>
            """
        ]

        for step in history:
            html_content.append(f'<div class="step-container">')
            html_content.append(f'<div class="step-header">Step {step["step"]} (t={step["t"]:.3f})</div>')
            html_content.append('<div class="sequence">')

            tokens = step["tokens"]
            is_gen = step["is_gen"]

            r_ins = step["rates"]["ins"]
            r_del = step["rates"]["del"]
            r_sub = step["rates"]["sub"]

            for i, token_str in enumerate(tokens):
                safe_token = html.escape(token_str).replace('Ġ', ' ').replace('Ċ', '⏎')

                if is_gen[i]:
                    val_ins = r_ins[i]
                    val_del = r_del[i]
                    val_sub = r_sub[i]

                    bg_color = f"rgba(255, 0, 0, {min(val_del * 2, 0.5)})"
                    border_style = f"2px solid rgba(0, 0, 255, {min(val_sub * 2, 1.0)})"
                    ins_width = f"{min(val_ins * 50, 100)}%"

                    tooltip = (f"Token: {safe_token}\n"
                               f"Ins Rate: {val_ins:.4f}\n"
                               f"Del Rate: {val_del:.4f}\n"
                               f"Sub Rate: {val_sub:.4f}")
                else:
                    bg_color = "#eee"
                    border_style = "1px solid #ccc"
                    ins_width = "0%"
                    tooltip = "Context (Fixed)"

                block = f"""
                <div class="token-box" style="background: {bg_color}; border-bottom: {border_style}">
                    <span class="token-text">{safe_token}</span>
                    <div class="indicator-bar"><div class="rate-ins" style="width: {ins_width}"></div></div>
                    <span class="tooltip">{tooltip}</span>
                </div>
                """
                html_content.append(block)

            html_content.append('</div></div>')

        html_content.append("</body></html>")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_content))

    def apply_ins_del_ops(self,
                          x_t: torch.Tensor,
                          ins_vals: torch.Tensor,
                          del_mask: torch.Tensor,
                          ) -> torch.Tensor:

        max_seq_len = self.config.model.length
        pad_token = self.pad_token
        mask_token = self.mask_token

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
            ins_b, ins_p = (ins_vals > 0).nonzero(as_tuple=True)

            num_insertions_at_loc = ins_vals[ins_b, ins_p].long()
            base_positions = new_pos[ins_b, ins_p]

            total_insertions = num_insertions_at_loc.sum()
            if total_insertions > 0:
                repeated_batch_indices = ins_b.repeat_interleave(num_insertions_at_loc)
                repeated_base_pos = base_positions.repeat_interleave(num_insertions_at_loc)

                flat_indices = torch.arange(total_insertions, device=device)

                group_starts = torch.cat([
                    torch.tensor([0], device=device),
                    num_insertions_at_loc.cumsum(dim=0)[:-1]
                ])

                repeated_starts = group_starts.repeat_interleave(num_insertions_at_loc)

                offsets = flat_indices - repeated_starts + 1

                final_ins_pos = repeated_base_pos + offsets

                valid_mask = (final_ins_pos >= 0) & (final_ins_pos < max_new_len)
                x_new[repeated_batch_indices[valid_mask], final_ins_pos[valid_mask]] = mask_token
        if max_new_len > max_seq_len:
            print(f"Warning: max_new_len {max_new_len} exceeds max_seq_len {max_seq_len}, truncating.")
            max_new_len = max_seq_len

        return x_new[:, :max_new_len]
