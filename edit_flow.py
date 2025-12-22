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
from flow_utils import rm_gap_tokens
from flows import CubicScheduler, x2prob, sample_p

LOG2 = math.log(2)


def fill_gap_tokens_with_repeats(
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


def sample_zt(z_0, z_1, mask_scheduler, default_scheduler, t, V, pad_token, gap_token, mask_token):
    z_neq = (z_0 != z_1) & (z_0 != pad_token) & (z_1 != pad_token)
    z_ins = (z_0 == gap_token) & (z_1 != gap_token) & z_neq  # (batch_size, z_seq_len)

    # t orig = (batch_size, 1) -> (batch_size, 1, 1)
    t = t.reshape(-1, 1, 1)

    mask_t = mask_scheduler(t)
    default_t = default_scheduler(t)

    # one-hot vecs (b, s, v)
    p_0 = x2prob(z_0, V + 4)
    p_1 = x2prob(z_1, V + 4)
    p_mask = x2prob(torch.tensor([mask_token]).expand_as(z_0).to(z_0.device), V + 4)

    # for insert
    pt_ins = (1 - mask_t) * p_0 \
             + mask_t * (1 - default_t) * p_mask \
             + mask_t * default_t * p_1

    # for delete/sub
    pt = (1 - default_t) * p_0 + default_t * p_1

    pt = torch.where(z_ins.unsqueeze(-1), pt_ins, pt)

    return sample_p(pt)


def poisson_make_uz_mask(
        z_t: torch.Tensor,
        z_1: torch.Tensor,
        pad_token,
        gap_token,
        vocab_size,
) -> torch.Tensor:
    """
    Create a mask for u_cat for indexing the output rate tensor based on differences between z_t and z_1.
    For each position i where z_t and z_1 differ, we index as follows:

    - z_t[i] = GAP_TOKEN & z_1[i] = c => u_mask[i, insert] = 1
    - z_t[i] = c & z_1[i] = GAP_TOKEN => u_mask[i, delete] = 1
    - z_t[i] = c1 & z_1[i] = c2 => u_mask[i, substitute, c1, c2] = 1
    """
    batch_size, z_seq_len = z_t.shape
    # n_ops = vocab_size + 2  # substitute + delete + insert

    z_neq = (z_t != z_1) & (z_t != pad_token) & (z_1 != pad_token)
    z_ins = (z_t == gap_token) & (z_1 != gap_token) & z_neq  # (batch_size, z_seq_len)
    z_del = (z_t != gap_token) & (z_1 == gap_token) & z_neq  # (batch_size, z_seq_len)
    z_sub = z_neq & ~z_ins & ~z_del  # (batch_size, z_seq_len)

    # mask (batch_size, z_seq_len, u_ops) where 1 indicates operation that bring z_t closer to z_1
    # u_mask = torch.zeros((batch_size, z_seq_len, n_ops), dtype=torch.bool, device=z_t.device)
    u_mask = torch.zeros((batch_size, z_seq_len, 3), dtype=torch.bool, device=z_t.device)

    # u_mask[z_ins, z_1[z_ins]] = True
    # u_mask[z_sub, z_1[z_sub]] = True

    u_mask[:, :, 0][z_sub] = True
    u_mask[:, :, 1][z_ins] = True
    u_mask[:, :, 2][z_del] = True

    assert z_neq.sum() == (z_ins | z_del | z_sub).sum(), "Mismatch in number of edits"
    assert z_neq.sum() == u_mask.sum(), "Mismatch in number of edits in mask"

    return u_mask


def _unsqueeze(x, reference):
    return x.view(
        *x.shape,
        *((1,) * (len(reference.shape) - len(x.shape))))


class NLL(torchmetrics.aggregation.MeanMetric):
    pass


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


class EditFlow(L.LightningModule):
    def __init__(
            self,
            config,
            tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.tokenizer = tokenizer

        self.V = self.tokenizer.vocab_size

        self.gen_ppl_eval_model_name_or_path = self.config.eval. \
            gen_ppl_eval_model_name_or_path

        if (not hasattr(self.tokenizer, 'mask_token')
                or self.tokenizer.mask_token is None):
            self.mask_token = self.V
            self.V += 1
        else:
            self.mask_token = self.tokenizer.mask_token_id

        self.pad_token = self.tokenizer.pad_token_id

        self.gap_token = 3  # todo hardcoded for now, use unused token or manually add one

        self.backbone = models.dit_edit_flow.DITEditFlow(
            self.config, vocab_size=self.V)

        # generative perplexity
        self.gen_ppl_metric = Perplexity()

        self.eval_model_tokenizer = transformers.AutoTokenizer. \
            from_pretrained(self.gen_ppl_eval_model_name_or_path)

        if self.eval_model_tokenizer.pad_token is None:
            self.eval_model_tokenizer.pad_token = \
                self.eval_model_tokenizer.eos_token
            self.eval_model_tokenizer.pad_token_id = \
                self.eval_model_tokenizer.eos_token_id

        self.lr = self.config.optim.lr

        self.time_conditioning = self.config.time_conditioning

        self.fast_forward_epochs = None
        self.fast_forward_batches = None

        # higher mass earlier for structure prediction, giving more time for unmasking
        self.mask_scheduler = CubicScheduler(a=3.0, b=0.0)

        # linear for unmasking (matches up with log linear sigma  = linear alpha with time from t = 1 to t = 0)
        self.default_scheduler = CubicScheduler(a=1.0, b=1.0)

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

    def _compute_loss(self, batch):
        x_0, x_1, z_0, z_1, t = batch

        z_t = self.sample_zt_sparse(z_0, z_1, t)

        x_t, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z_t, pad_token=self.pad_token, gap_token=self.gap_token)

        u_t, sub_logits = self.backbone.forward(x_t, t, x_pad_mask)
        u_tot = u_t.sum(dim=(1, 2))

        uz_mask = poisson_make_uz_mask(z_t, z_1, vocab_size=self.V,
                                       gap_token=self.gap_token, pad_token=self.pad_token)

        target_sub = uz_mask[:, :, 0]
        target_ins = uz_mask[:, :, 1]
        target_del = uz_mask[:, :, 2]

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

        default_coeff = (self.default_scheduler.derivative(t) / (1 - self.default_scheduler(t) + eps)).squeeze()
        ins_coeff = (self.mask_scheduler.derivative(t) / (1 - self.mask_scheduler(t) + eps)).squeeze()
        z_ins_event = (z_0 == self.gap_token) & (z_1 != self.gap_token) & (z_0 != z_1)
        sched_coeff = torch.where(z_ins_event, ins_coeff.unsqueeze(-1), default_coeff.unsqueeze(-1))


        term2 = (selected_log_ll * sched_coeff).sum(dim=1)
        loss = u_tot - term2

        u_ins = u_t[:, :, 0].sum(dim=1).mean()
        u_del = u_t[:, :, 2].sum(dim=1).mean()
        u_sub = u_t[:, :, 1].sum(dim=1).mean()

        return loss.mean(), u_tot, u_ins, u_del, u_sub, term2.mean()


    def sample_zt_sparse(self, z_0, z_1, t):
        t = t.reshape(-1, 1) # [Batch, 1]
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
            # while t.max() <= 1 - default_h:
            while t.max() <= 1:
                u_t, sub_logits = self.backbone.forward(x_t, t, x_pad_mask)
                sub_probs = F.softmax(sub_logits, dim=-1)

                lambda_ins = u_t[:, :, 0]  # Insertion rate        (n_samples, x_seq_len)
                lambda_sub = u_t[:, :, 1]  # Substitution rate     (n_samples, x_seq_len)
                lambda_del = u_t[:, :, 2]  # Deletion rate         (n_samples, x_seq_len)

                # print(f"Lambda Stats: t = {t.mean().item():.4f}, Min={lambda_ins.min().item():.4f}, Max={lambda_ins.max().item():.4f}, Mean: {lambda_ins.mean().item()}")

                # print (f'lam_shape: {lambda_ins[0].shape}')
                # print (f'x_shape: {x_t.shape}')

                # print (f'num_valid_tokens: {valid_token_mask.sum(dim=1)}')
                valid_token_mask = (~x_pad_mask).float()

                lambda_ins = lambda_ins * valid_token_mask
                lambda_sub = lambda_sub * valid_token_mask
                lambda_del = lambda_del * valid_token_mask

                # adapt_h = get_adaptive_h(default_h, t, scheduler)
                adapt_h = default_h

                ins_vals = torch.poisson(adapt_h * lambda_ins).long()
                # ins_vals = torch.poisson(get_analytic_mean(lambda_ins, t, adapt_h)).long()

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
