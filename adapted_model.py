import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
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


def create_random_mask(attn_ratio, seq_len):
    random_mask = (torch.rand(seq_len, seq_len) < attn_ratio).to('cuda')

    def random_mask_func(b, h, q_idx, kv_idx):
        return random_mask[q_idx][kv_idx]

    return random_mask_func



class AdaptedEditFlowsTransformer(nn.Module):
    def __init__(self, pretrained_model_name: str, hidden_dim=512):
        super().__init__()

        # godel_id = "Goedel-LM/Goedel-Prover-V2-8B"
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, dtype=torch.bfloat16,
                                                          trust_remote_code=True,
                                                          _attn_implementation='flex_attention').train()

        self.model.compile()

        self.vocab_size = self.model.config.vocab_size
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.rate_head = nn.Sequential(nn.Linear(self.model.config.hidden_size + hidden_dim, hidden_dim), nn.SiLU(),
                                       nn.Linear(hidden_dim, 3))
        self.ins_head = nn.Sequential(nn.Linear(self.model.config.hidden_size + hidden_dim, hidden_dim), nn.SiLU(),
                                      nn.Linear(hidden_dim, self.vocab_size))
        self.sub_head = nn.Sequential(nn.Linear(self.model.config.hidden_size + hidden_dim, hidden_dim), nn.SiLU(),
                                      nn.Linear(hidden_dim, self.vocab_size))
        self._init_heads()

    def _init_heads(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor, t: torch.Tensor, pad_mask: torch.Tensor, attn_mask_ratio: float = 0.0):
        B, L = tokens.shape

        # todo
        # padding mask to the model might only be needed for inference, since it defaults to 0 gradient and is masked out from the loss
        # padding_mask = create_padding_mask(pad_mask)

        if attn_mask_ratio < 1.0:
            #compute block for annealed attention
            random_func = create_random_mask(attn_mask_ratio, L)
            or_mask = or_masks(*[causal_mask, random_func])
            final_mask = or_mask
            # final_mask = and_masks(*[or_mask, padding_mask])
        else:
            # keep full attention
            # final_mask = padding_mask
            final_mask = full_mask

        block_mask = create_block_mask(final_mask, None, None, L, L, device='cuda')  # , _compile=True)

        # print(tokens.shape)

        outputs = self.model.forward(input_ids=tokens, attention_mask=block_mask, output_hidden_states=True,
                                     kernel_options={
                                         "BLOCK_M": 32,
                                         "BLOCK_N": 32,
                                         "BLOCK_M1": 32,
                                         "BLOCK_N1": 32,
                                         "BLOCK_M2": 32,
                                         "BLOCK_N2": 32,
                                     })

        hidden_states = outputs.hidden_states[-1]

        time_ = self.time_emb(t).unsqueeze(1).expand(-1, L, -1)

        x = torch.cat([hidden_states, time_], dim=-1)

        rates = F.softplus(self.rate_head(x))
        ins = F.softmax(self.ins_head(x), dim=-1)
        sub = F.softmax(self.sub_head(x), dim=-1)

        mask = (~pad_mask).unsqueeze(-1).float()

        return (rates * mask, ins * mask, sub * mask)
