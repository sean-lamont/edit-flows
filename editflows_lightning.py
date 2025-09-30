# use attention mask below with edit flow model
# #### full attention
#         attn_mask_ratio = min(1.0, (self.state.global_step + 1) / self.diff_args.anneal_steps)
#         # attn_mask_ratio = 1.0
#         x_embed = get_embeds(x)
#
#         attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=x_embed.dtype, device=x.device, attn_mask_ratio=attn_mask_ratio)
#
#         # attention_mask = torch.ones_like(x_t, dtype=torch.float)
#         logits = model(x_t, attention_mask=attention_mask)
#
#         loss_mask = x_t == self.tokenizer.mask_token_id
#
#         if self.diff_args.shift:
#             #### shift loss
#             logits = logits[:,:-1]
#             loss_mask = loss_mask[:,1:]
#             x = x[:,1:]


# - Have model class which takes pre-trained model, adds heads, and sets up forward function to return required rates, ins, sub logits


import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl


class EditFlowAdaptedModel(nn.Module):
    def __init__(self, base_model, tokenizer, diff_args):
        super(EditFlowAdaptedModel, self).__init__()
        self.base_model = base_model

        hidden_size = base_model.config.hidden_size
        vocab_size = base_model.config.vocab_size

        # Heads for predicting insertion and substitution logits
        self.insertion_head = nn.Linear(hidden_size, vocab_size)
        self.substitution_head = nn.Linear(hidden_size, vocab_size)

        self.rates_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 3),  # Output 3 rates (insert, substitute, delete)
        )
        self.ins_logits_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, vocab_size),  # Output vocab_size insert probabilities
        )
        self.sub_logits_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, vocab_size),  # Output vocab_size substitute probabilities
        )

    def forward(self, tokens: T["batch", "x_seq_len", "long"],
                time_step: T["batch", 1, "float"],
                padding_mask: T["batch", "x_seq_len", "bool"],
                attn_mask_ratio: float = 1.0) -> Tuple[
        T["batch", "x_seq_len", "float"],  # Rates (3 values)
        T["batch", "x_seq_len", "vocab_size"],  # Insert probabilities (vocab_size values)
        T["batch", "x_seq_len", "vocab_size"],  # Substitute probabilities (vocab_size values)
    ]:
        """Forward pass takes in x_t, t, and padding mask, returns rates and probabilities
        """

        batch_size, seq_len = tokens.shape

        attention_mask = get_anneal_attn_mask(seq_len, batch_size, dtype=x_embed.dtype, device=tokens.device,
                                              attn_mask_ratio=attn_mask_ratio)

        outputs = self.base_model(input_ids=tokens, attention_mask=attention_mask).logits

        ins_logits = self.ins_logits_out(outputs)  # (batch_size, seq_len, vocab_size)
        sub_logits = self.sub_logits_out(outputs)  # (batch_size, seq_len, vocab_size)
        rates = F.softplus(self.rates_out(outputs))  # (batch_size, seq_len, 3) - ensure positive rates

        ins_probs = F.softmax(ins_logits, dim=-1)  # (batch_size, seq_len, vocab_size)
        sub_probs = F.softmax(sub_logits, dim=-1)  # (batch_size, seq_len, vocab_size)

        # Zero out outputs for padded positions
        mask_expanded = (~padding_mask).unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        rates = rates * mask_expanded
        ins_probs = ins_probs * mask_expanded
        sub_probs = sub_probs * mask_expanded

        if torch.isnan(rates).any() or torch.isnan(ins_probs).any() or torch.isnan(sub_probs).any():
            raise ValueError("NaN detected in output probabilities or rates")

        return (
            cast(T["batch", "seq_len", "float"], rates),
            cast(T["batch", "seq_len", "vocab_size"], ins_probs),
            cast(T["batch", "seq_len", "vocab_size"], sub_probs),
        )


def get_anneal_attn_mask(seq_len, bsz, dtype, device, attn_mask_ratio):
    mask = torch.full((seq_len, seq_len), 0, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 1)
    causal_mask = mask.to(dtype)

    random_mask = torch.bernoulli(torch.full((seq_len, seq_len), 0.0, device=device) + attn_mask_ratio)

    anneal_mask = torch.logical_or(causal_mask, random_mask)
    expanded_mask = anneal_mask[None, None, :, :].expand(bsz, 1, seq_len, seq_len)
    inverted_mask = 1.0 - expanded_mask.to(dtype)

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)
