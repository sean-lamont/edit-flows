import torch
import time
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from utils import SinusoidalTimeEmbedding
from torch.nn.attention.flex_attention import create_block_mask, and_masks, or_masks, create_mask


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


# bit of a hack, but can't just return True
def full_mask(b, h, q_idx, kv_idx):
    return q_idx >= 0


def create_padding_mask(pads):
    def padding(b, h, q_idx, kv_idx):
        return ~pads[b, q_idx] & ~pads[b, kv_idx]

    return padding


def create_random_mask(attn_ratio, seq_len, device='cuda'):
    random_mask = (torch.rand(seq_len, seq_len) < attn_ratio).to(device)

    def random_mask_func(b, h, q_idx, kv_idx):
        return random_mask[q_idx][kv_idx]

    return random_mask_func


class AdaptedEditFlowsTransformer(nn.Module):
    def __init__(self, pretrained_model_name: str, hidden_dim=512):
        super().__init__()

        # bnb_conf = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )
        # godel_id = "Goedel-LM/Goedel-Prover-V2-8B"
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, dtype=torch.bfloat16,
                                                          trust_remote_code=True,
                                                          _attn_implementation='flex_attention',
                                                          # _attn_implementation='flash_attention_2', # _attn_implementation='flex_attention',
                                                          # quantization_config=bnb_conf,
                                                          ).train()
        #
        # self.model = prepare_model_for_kbit_training(self.model)

        # add lora and quantization to model:

        # add LoRa and Quantization

        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
            lora_dropout=0.05,
            # bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        self.model.gradient_checkpointing_enable()

        self.model.compile()

        self.vocab_size = self.model.config.vocab_size
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)

        self.rate_head = nn.Sequential(nn.Linear(self.model.config.hidden_size + hidden_dim, hidden_dim * 2), nn.SiLU(),
                                       nn.Linear(hidden_dim * 2, 5)) # 3 for ins,sub,del, extra 2 for weighting lm_head

        self.ins_head = nn.Sequential(nn.Linear(self.model.config.hidden_size + hidden_dim, self.model.config.hidden_size), nn.SiLU(),
                                      nn.Linear(self.model.config.hidden_size, self.vocab_size))

        self.sub_head = nn.Sequential(nn.Linear(self.model.config.hidden_size + hidden_dim, self.model.config.hidden_size), nn.SiLU(),
                                      nn.Linear(self.model.config.hidden_size, self.vocab_size))

        # self._init_heads()

    # def _init_heads(self):
    #     for m in self.rate_head.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight, gain=0.1)
    #             if m.bias is not None: nn.init.zeros_(m.bias)
    #     for m in self.sub_head.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight, gain=0.1)
    #             if m.bias is not None: nn.init.zeros_(m.bias)
    #     for m in self.ins_head.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight, gain=0.1)
    #             if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor, t: torch.Tensor, pad_mask: torch.Tensor, context_tokens, pad_token,
                attn_mask_ratio: float = 0.0):

        # x0 = torch.cat((context_ids.squeeze(0), prev_ids.squeeze(0)[1:]), dim=0)[:self.max_len]

        context_lens = [c.shape[0] for c in context_tokens]
        pad_len = tokens.shape[1] + max(context_lens)

        combined_tokens = torch.stack([
            F.pad(
                torch.cat([context_tokens[i], tokens[i]], dim=0),
                (0, pad_len - (tokens.shape[1] + context_lens[i])), value=pad_token)
            for i in range(tokens.shape[0])], dim=0).long()

        B, L = combined_tokens.shape

        padding_mask = create_padding_mask(combined_tokens == pad_token)

        if attn_mask_ratio < 1.0:
            # compute block for annealed attention
            random_func = create_random_mask(attn_mask_ratio, L, device=tokens.device)
            or_mask = or_masks(*[causal_mask, random_func])
            # final_mask = or_mask
            final_mask = and_masks(*[or_mask, padding_mask])
        else:
            # keep full attention
            final_mask = padding_mask
            # final_mask = full_mask


        block_mask = create_block_mask(final_mask, B, None, L, L, device=tokens.device)  # , _compile=True)

        # block_mask = create_block_mask(final_mask, None, None, L, L, device=tokens.device)  # , _compile=True)
        # outputs = self.model.forward(input_ids=tokens,  output_hidden_states=True,)

        outputs = self.model.forward(input_ids=combined_tokens, attention_mask=block_mask, output_hidden_states=True,
                                     kernel_options={
                                         "BLOCK_M": 32,
                                         "BLOCK_N": 32,
                                         "BLOCK_M1": 32,
                                         "BLOCK_N1": 32,
                                         "BLOCK_M2": 32,
                                         "BLOCK_N2": 32,
                                     })

        # (b, seq_len, dim)
        hidden_states = outputs.hidden_states[-1]

        # only take hidden states from context_lens onwards
        hidden_states = torch.stack([
            hidden_states[i, context_lens[i]:context_lens[i] + tokens.shape[1]]  # same length (original padded non-context tokens, taken from respective context )
            for i in range(hidden_states.shape[0])], dim=0 )

        time_ = self.time_emb(t).unsqueeze(1).expand(-1, hidden_states.shape[1], -1)

        x = torch.cat([hidden_states, time_], dim=-1)

        # (b, seq, 5)
        rates = F.softplus(self.rate_head(x))

        # (b, seq, vocab)
        lm_output = self.model.lm_head(hidden_states)

        # print (lm_output.shape)

        # add zero vector for first entry
        lm_output = torch.cat([torch.zeros_like(lm_output[:, 0], device=x.device, dtype=torch.bfloat16).unsqueeze(1), lm_output], dim=1)

        # add lm output for insert (usual objective from pretrained model)
        ins = F.softmax(self.ins_head(x) + rates[:, :, -1].unsqueeze(-1) * lm_output[:, 1:], dim=-1)

        sub = F.softmax(self.sub_head(x) + rates[:, :, -2].unsqueeze(-1) * lm_output[:, :-1], dim=-1)

        mask = (~pad_mask).unsqueeze(-1).float()

        return (rates[:, :, :3] * mask, ins * mask, sub * mask)
