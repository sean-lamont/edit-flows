import torch
from utils import PAD_TOKEN
import torch.nn as nn
import torch.nn.functional as F
from utils import SinusoidalTimeEmbedding


class SimpleEditFlowsTransformer(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim=512, num_layers=6, num_heads=8, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_dim)
        self.pad_token = PAD_TOKEN
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim*4,
                                       activation='gelu', batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.rate_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 3))
        self.ins_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, vocab_size))
        self.sub_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, vocab_size))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None: nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens: torch.Tensor, t: torch.Tensor, pad_mask: torch.Tensor):
        B, L = tokens.shape
        tok = self.token_emb(tokens)
        pos = self.pos_emb(torch.arange(L, device=tokens.device)).unsqueeze(0)
        time = self.time_emb(t).unsqueeze(1).expand(-1, L, -1)
        x = tok + pos + time
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        rates = F.softplus(self.rate_head(x))
        ins = F.softmax(self.ins_head(x), dim=-1)
        sub = F.softmax(self.sub_head(x), dim=-1)
        mask = (~pad_mask).unsqueeze(-1).float()
        return (rates * mask, ins * mask, sub * mask)