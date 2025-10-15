import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import transformers
from utils import SinusoidalTimeEmbedding
from adapt_qwen import qwen_new_forward



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




class AdaptedEditFlowsTransformer(nn.Module):
    def __init__(self, pretrained_model_name: str, hidden_dim=512):
        super().__init__()

        transformers.models.qwen2.modeling_qwen3.Qwen3Model.forward = qwen_new_forward
        # godel_id = "Goedel-LM/Goedel-Prover-V2-8B"
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)


        self.vocab_size = self.model.config.vocab_size
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.rate_head = nn.Sequential(nn.Linear(self.model.config.hidden_size + hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))
        self.ins_head = nn.Sequential(nn.Linear(self.model.config.hidden_size + hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.vocab_size))
        self.sub_head = nn.Sequential(nn.Linear(self.model.config.hidden_size + hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, self.vocab_size))
        self._init_heads()

    def _init_heads(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, tokens: torch.Tensor, t: torch.Tensor, pad_mask: torch.Tensor, attn_mask_ratio: float = 0.0):
        B, L = tokens.shape

        anneal_mask = get_anneal_attn_mask(L, B, dtype=self.model.dtype, device=tokens.device, attn_mask_ratio=attn_mask_ratio)


        # update the 4D anneal mask with pad_mask to account for padding tokens, so nothing attends to pads. Note that pad_mask is True for pad tokens
        anneal_mask = anneal_mask.masked_fill(pad_mask[:, None, None, :], torch.finfo(self.model.dtype).min)


        outputs = self.model(input_ids=tokens, attention_mask=anneal_mask, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[-1]
        
        time = self.time_emb(t).unsqueeze(1).expand(-1, L, -1)
        
        x = torch.cat([hidden_states, time], dim=-1)

        rates = F.softplus(self.rate_head(x))
        ins = F.softmax(self.ins_head(x), dim=-1)
        sub = F.softmax(self.sub_head(x), dim=-1)
        
        mask = (~pad_mask).unsqueeze(-1).float()
        return (rates * mask, ins * mask, sub * mask)
