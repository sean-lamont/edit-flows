import pytorch_lightning as L
import torch
import torch.nn.functional as F
import transformers


class LLaDAFineTune(L.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.tokenizer = tokenizer

        print(f"Loading LLaDA Backbone: {config.model_name}")
        self.backbone = transformers.AutoModel.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

        self.mask_token_id = 126336

        if config.get('freeze_backbone', False):
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        return self.backbone(input_ids, attention_mask=attention_mask).logits

    def forward_process(self, input_ids, context_mask, mask_prob=0.15):
        """
        Applies linear noise schedule (masking) respecting the context mask.
        """
        b, l = input_ids.shape
        device = input_ids.device

        t = torch.rand(b, device=device)

        p_mask = (1 - mask_prob) * t + mask_prob
        p_mask = p_mask[:, None].expand(b, l)

        is_masked = torch.rand((b, l), device=device) < p_mask

        # context_mask is 1 for prompt, 0 for code. We only mask where context_mask is 0.
        is_masked = is_masked & (~context_mask)

        noisy_input_ids = input_ids.clone()
        noisy_input_ids[is_masked] = self.mask_token_id

        return noisy_input_ids, is_masked

    def training_step(self, batch, batch_idx):
        # input_ids: [PAD]... [Prompt] [Code]
        # context_mask: 1 for PAD/Prompt, 0 for Code
        data = batch
        input_ids = data['input_ids']
        context_mask = data['context_mask']

        noisy_input, masked_indices = self.forward_process(input_ids, context_mask)

        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        logits = self(noisy_input, attention_mask=attention_mask)

        loss = F.cross_entropy(
            logits[masked_indices],
            input_ids[masked_indices],
            reduction='mean'
        )

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch
        input_ids = data['input_ids']
        context_mask = data['context_mask']

        prompt_len = context_mask.sum(dim=1)

        if batch_idx == 0:
            self._log_generation(input_ids, context_mask)

        return None  # Placeholder for metrics

    def _log_generation(self, input_ids, context_mask):
        # Take first sample
        idx = 0
        full_seq = input_ids[idx]
        mask = context_mask[idx]

        prompt_ids = full_seq[mask]
        gt_code_ids = full_seq[~mask]

        prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        gt_code = self.tokenizer.decode(gt_code_ids, skip_special_tokens=True)

        prompt_tensor = prompt_ids.unsqueeze(0)

        gen_len = self.config.get('val_gen_len', 128)

        generated_ids = self.generate(
            prompt_tensor,
            steps=self.config.get('sampling_steps', 64),
            gen_length=gen_len
        )

        # Extract generated part
        gen_seq = generated_ids[0, prompt_tensor.shape[1]:]
        gen_code = self.tokenizer.decode(gen_seq, skip_special_tokens=True)

        print("\n" + "=" * 40)
        print(f"VAL SAMPLE (Step {self.global_step})")
        print(f"PROMPT: {prompt_str[:100]}...")
        print(f"GT CODE: {gt_code[:100]}...")
        print(f"GEN CODE: {gen_code[:100]}...")
        print("=" * 40 + "\n")

    @torch.no_grad()
    def generate(self, prompt, steps=64, gen_length=128, block_length=128,
                 temperature=0., cfg_scale=0., remasking='low_confidence'):
        """
        Adapted from LLaDA official inference code.
        """
        model = self.backbone
        mask_id = self.mask_token_id

        x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        block_length = min(block_length, gen_length)
        num_blocks = gen_length // block_length
        if num_blocks == 0: num_blocks = 1

        steps = steps // num_blocks

        for num_block in range(num_blocks):
            # Define window for this block
            block_start = prompt.shape[1] + num_block * block_length
            block_end = prompt.shape[1] + (num_block + 1) * block_length

            block_mask_index = (x[:, block_start:block_end] == mask_id)

            # Linear Schedule: How many tokens to unmask per step
            num_transfer_tokens = self._get_num_transfer_tokens(block_mask_index, steps)

            for i in range(steps):
                mask_index = (x == mask_id)

                attention_mask = torch.ones_like(x)

                logits = model(x, attention_mask=attention_mask).logits

                logits_with_noise = self._add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                else:
                    x0_p = torch.rand_like(x0, dtype=torch.float)

                x0_p[:, block_end:] = -float('inf')

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -float('inf'))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    k = num_transfer_tokens[j, i]
                    if k > 0:
                        _, select_index = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_index] = True

                x[transfer_index] = x0[transfer_index]

        return x

    def _get_num_transfer_tokens(self, mask_index, steps):
        """Precomputes how many tokens to unmask at each step (Linear Schedule)."""
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps

        num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.long) + base
        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, :remainder[i]] += 1
        return num_transfer_tokens

    def _add_gumbel_noise(self, logits, temperature):
        if temperature == 0: return logits
        noise = torch.rand_like(logits)
        gumbel_noise = (-torch.log(noise + 1e-10)) ** temperature
        return logits.exp() / (gumbel_noise + 1e-10)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config.lr, weight_decay=0.01)
