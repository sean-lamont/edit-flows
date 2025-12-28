import torch
# set up lora with all linear layers:
from peft import LoraConfig, get_peft_model
from sacrebleu import corpus_bleu
import torch.nn as nn
import torch.nn.functional as F
import transformers
from tqdm import tqdm

from edit_flow import EditFlow


class LLaDABackbone(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config

        print(f"Loading LLaDA Backbone")

        # bnb_config = transformers.BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",  # NormalFloat4 is best for pre-trained weights
        #     bnb_4bit_use_double_quant=True,  # Compresses the quantization constants
        #     bnb_4bit_compute_dtype=torch.bfloat16  # Compute in bf16 for stability
        # )

        self.base_model = transformers.AutoModel.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct",
            trust_remote_code=True,
            # quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=None
        )

        self.base_model = get_peft_model(self.base_model, lora_config)
        self.base_model.print_trainable_parameters()

        self.hidden_size = self.base_model.config.hidden_size
        self.vocab_size = vocab_size

        # Heads for Edit Flow
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

        # Re-use the LM Head
        self.content_head = self.base_model.model.transformer.ff_out

        # Unfreeze heads
        for param in self.content_head.parameters():
            param.requires_grad = True

    # todo add time conditioning?
    def forward(self, x, t=None, attention_mask=None):
        """
        Forward pass assuming Right Padding.
        attention_mask: Bool [B, L], True=Pad (Ignore), False=Content (Attend)
        """
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):

            # Convert internal pad mask (True=Pad) to HF attention mask (1=Attend)
            if attention_mask is not None:
                hf_mask = (~attention_mask).long()
            else:
                hf_mask = None

            outputs = self.base_model(x, attention_mask=hf_mask, output_hidden_states=True)
            h = outputs.hidden_states[-1]

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

        self.mask_token_id = 126336
        self.backbone = LLaDABackbone(self.config, vocab_size=self.V)
        self.gap_token = 126084

        self.tokenizer.padding_side = 'right'

        # safety check
        self.tokenizer.pad_token_id = 126085
        self.pad_token = 126085
        # if self.tokenizer.pad_token_id is None:
        #     self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _compute_loss(self, batch):
        x_0, x_1, z_0, z_1, t, context_mask = batch
        bsz = x_0.shape[0]

        z_t = self.sample_zt(z_0, z_1, t)

        z_t = torch.where(context_mask, z_1, z_t)

        x_t, x_pad_mask, z_gap_mask, z_pad_mask = self.rm_gap_tokens(z_t)

        u_t_logits, sub_vocab_logits = self.backbone.forward(x_t, t, x_pad_mask)

        sched_coeff_z = self.get_sched_coeff(t, z_0, z_1, z_t)

        ignore_mask = (x_pad_mask | context_mask)  # True for Pad

        raw_sub = u_t_logits[:, :, 0]
        raw_ins = u_t_logits[:, :, 1]
        raw_del = u_t_logits[:, :, 2]

        sub_vocab_logits = sub_vocab_logits.masked_fill(ignore_mask.unsqueeze(-1), -1e9)

        if self.time_dependent:
            r_sub = F.softplus(torch.clamp(raw_sub, max=1e6)).masked_fill(ignore_mask, 0)
            r_ins = F.softplus(torch.clamp(raw_ins, max=1e6)).masked_fill(ignore_mask, 0)
            r_del = F.softplus(torch.clamp(raw_del, max=1e6)).masked_fill(ignore_mask, 0)

            u_tot = (r_ins + r_sub + r_del).sum(dim=-1)

            eps = 1e-9
            log_r_sub = torch.log(r_sub + eps)
            log_r_ins = torch.log(r_ins + eps)
            log_r_del = torch.log(r_del + eps)

        else:  # Time Independent
            r_ins = F.softplus(raw_ins).masked_fill(ignore_mask, 0)
            r_sub = torch.sigmoid(raw_sub).masked_fill(ignore_mask, 0)
            r_del = torch.sigmoid(raw_del).masked_fill(ignore_mask, 0)

            if self.rate_scaling:
                sched_coeff_x = torch.zeros_like(r_ins)

                if sched_coeff_z.dim() > 1:
                    mask_z = ~z_gap_mask
                    ranks = mask_z.cumsum(dim=1) - 1
                    valid_z = mask_z & (ranks < x_t.shape[1])

                    values = sched_coeff_z[valid_z]
                    dest_cols = ranks[valid_z]
                    dest_rows = torch.arange(bsz, device=x_t.device).unsqueeze(1).expand_as(z_t)[valid_z]

                    sched_coeff_x[dest_rows, dest_cols] = values.to(dtype=sched_coeff_x.dtype)
                else:
                    sched_coeff_x = sched_coeff_z

                u_tot = (r_ins * sched_coeff_x).sum(dim=-1) + \
                        (r_sub * sched_coeff_x).sum(dim=-1) + \
                        (r_del * sched_coeff_x).sum(dim=-1)
            else:
                u_tot = r_ins.sum(dim=-1) + r_sub.sum(dim=-1) + r_del.sum(dim=-1)

            log_r_ins = torch.log(r_ins + 1e-9)
            log_r_sub = F.logsigmoid(raw_sub).masked_fill(ignore_mask, -1e9)
            log_r_del = F.logsigmoid(raw_del).masked_fill(ignore_mask, -1e9)

        # 5. Map Predictions back to Z-Space for Loss
        lse_sub_x = sub_vocab_logits.logsumexp(dim=-1)
        packed_features_x = torch.stack([log_r_sub, log_r_ins, log_r_del, lse_sub_x], dim=-1)

        packed_features_z = self.fill_gap_tokens_with_repeats(
            packed_features_x, z_gap_mask, z_pad_mask
        )

        log_rate_sub_z = packed_features_z[..., 0]
        log_rate_ins_z = packed_features_z[..., 1]
        log_rate_del_z = packed_features_z[..., 2]
        lse_sub_z = packed_features_z[..., 3]

        uz_mask = self.make_uz_mask(z_t, z_1)
        sub_mask = uz_mask[:, :, 0] & ~context_mask
        ins_mask = uz_mask[:, :, 1] & ~context_mask
        del_mask = uz_mask[:, :, 2] & ~context_mask

        # 6. Gather Targets (Z -> X indices)
        # Right Padded: Index in X = Rank of non-gap token in Z
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

    @torch.no_grad()
    def _sample_conditional(self, batch, n_steps=None, eps=1e-5):
        """
        Conditional Sampling assuming RIGHT PADDING everywhere.
        """
        x_0_in, _, _, _, _, _ = batch
        bsz = x_0_in.shape[0]

        max_capacity = self.config.model.length
        device = self.device

        # Fix Left Padding if detected (Robustness check)
        if (x_0_in[:, 0] == self.pad_token).all() and (x_0_in[:, -1] != self.pad_token).any():
            x_0_right = torch.full_like(x_0_in, self.pad_token)
            lens = (x_0_in != self.pad_token).sum(dim=1)
            for i in range(bsz):
                l = lens[i].item()
                if l > 0: x_0_right[i, :l] = x_0_in[i, -l:]
            x_0_in = x_0_right

        if n_steps is None: n_steps = self.config.sampling.steps
        default_h = 1 / n_steps
        t = torch.ones(bsz, 1, device=device) * 0.01

        # Initialize State
        x_t = x_0_in.clone()
        context_lens = (x_0_in != self.pad_token).sum(dim=1)
        valid_seq_lens = context_lens.clone()

        with tqdm(desc="Euler Sampling", total=n_steps) as pbar:
            while t.max() <= 1:

                current_len = x_t.shape[1]

                x_pad_mask = (x_t == self.pad_token)

                # Construct precise attention mask using CURRENT length
                seq_range = torch.arange(current_len, device=device).unsqueeze(0)
                is_valid_range = seq_range < valid_seq_lens.unsqueeze(1)

                # Pad Mask for Model: True=Pad/Ignore.
                model_pad_mask = ~is_valid_range

                # Forward (Shapes now match: x_t [B, L], mask [B, L])
                u_t, sub_logits = self.backbone.forward(x_t, t, x_pad_mask)

                # 3. Compute Rates
                if model_pad_mask.any():
                    sub_logits[model_pad_mask] = 0.0

                sub_probs = F.softmax(sub_logits, dim=-1)
                sub_probs = sub_probs.masked_fill(model_pad_mask, 0)

                lambda_sub = u_t[:, :, 0]
                lambda_ins = u_t[:, :, 1]
                lambda_del = u_t[:, :, 2]

                if not self.time_dependent:
                    lambda_sub = torch.sigmoid(lambda_sub)
                    lambda_del = torch.sigmoid(lambda_del)
                    lambda_ins = F.softplus(torch.clamp(lambda_ins, max=1e6))
                else:
                    lambda_ins = F.softplus(torch.clamp(lambda_ins, max=1e6))
                    lambda_sub = F.softplus(torch.clamp(lambda_sub, max=1e6))
                    lambda_del = F.softplus(torch.clamp(lambda_del, max=1e6))

                if not self.time_dependent:  # scale raw count/bernoulli predictions by sampler rate
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

                # subtract 1 to keep single token for gen
                is_context = seq_range < context_lens.unsqueeze(1) - 1

                is_gen = is_valid_range & ~is_context
                # is_boundary = (seq_range == valid_seq_lens.unsqueeze(1)) # Removed

                lambda_sub = torch.where(is_gen, lambda_sub, 0.0)
                lambda_del = torch.where(is_gen, lambda_del, 0.0)
                lambda_ins = torch.where(is_gen, lambda_ins, 0.0)

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

                # 6. Update Valid Lengths
                total_ins = ins_vals.sum(dim=1)
                total_del = del_mask.long().sum(dim=1)
                valid_seq_lens = valid_seq_lens + total_ins - total_del

                valid_seq_lens = valid_seq_lens.clamp(min=context_lens, max=torch.tensor(max_capacity, device=device))

                t += adapt_h
                pbar.update(1)

        return x_t

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.sample_batches = []
        if batch_idx < self.config.sampling.num_sample_batches:
            self.sample_batches.append(batch)

        loss, u_tot, u_ins, u_del, u_sub, term2 = self._compute_loss(batch)
        self.log_dict({
            "val_loss": loss,
            "val_u_tot": u_tot,
            "val_u_ins": u_ins,
            "val_u_del": u_del,
            "val_u_sub": u_sub,
        }, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self):
        # Keeps original logging
        if ((self.config.eval.compute_perplexity_on_sanity
             or not self.trainer.sanity_checking)
                and self.config.eval.generate_samples):

            all_gen_lens = []
            samples, text_samples = None, None
            for i in range(len(self.sample_batches)):
                samples = self._sample_conditional(self.sample_batches[i])

                # Calculate non-padded lengths
                batch_lens = (samples != self.tokenizer.pad_token_id).sum(dim=1).float()
                all_gen_lens.append(batch_lens)

                samples = [s[s != self.tokenizer.pad_token_id] for s in samples]

                # Decode the samples to be re-tokenized by eval model
                text_samples = self.tokenizer.batch_decode(samples)

                # rather than old ppl eval, just check bleu score to ground truth x1
                x_1_batch = self.sample_batches[i][1]
                x_1_batch = [s[s != self.tokenizer.pad_token_id] for s in x_1_batch]
                text_x_1_batch = self.tokenizer.batch_decode(x_1_batch)

                # Compute BLEU score
                bleu_score = self.compute_bleu_score(text_samples, text_x_1_batch)
                self.log("val_bleu_score", bleu_score, on_epoch=True, prog_bar=True, sync_dist=True)

            #     if self.config.eval.compute_generative_perplexity:
            #         self.compute_generative_perplexity(text_samples)
            #
            # if all_gen_lens:
            #     all_gen_lens = torch.cat(all_gen_lens)
            #     self.log_dict({
            #         "val/gen_len_mean": all_gen_lens.mean(),
            #         "val/gen_len_std": all_gen_lens.std(),
            #     }, prog_bar=True, on_epoch=True, sync_dist=True)

            if self.trainer.global_rank == 0 and hasattr(self.trainer.logger, 'log_table'):
                # Log the last generated samples
                text_samples = text_samples[:self.config.sampling.num_sample_log]

                for sample in text_samples:
                    print('Sample: ')
                    print(sample)
                    print('\n')

                self.trainer.logger.log_table(
                    key=f'samples@global_step{self.global_step}',
                    columns=['Generated Samples'],
                    data=[[s] for s in text_samples])

    def compute_bleu_score(self, samples, ground_truth):
        # sacrebleu expects a list of references for each hypothesis, so wrap each ground truth in a list
        references = [[gt] for gt in ground_truth]
        bleu = corpus_bleu(samples, references)
        return bleu.score
