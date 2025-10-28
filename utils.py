from typing import Tuple, List

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half = self.dim // 2
        freq = torch.exp(-torch.arange(half, device=t.device, dtype=t.dtype) *
                         torch.log(torch.tensor(10000.0, device=t.device, dtype=t.dtype)) / (half - 1))
        phase = t * freq
        emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.proj(emb)

def x2prob(x: Tensor, vocab_size: int) -> Tensor:
    return torch.nn.functional.one_hot(x, num_classes=vocab_size).float()

def sample_p(pt: Tensor) -> Tensor:
    b, L, C = pt.shape
    flat = rearrange(pt, 'b l c -> (b l) c')
    idx = torch.multinomial(flat, 1).view(b, L)
    return idx

def sample_cond_pt(p0: Tensor, p1: Tensor, t: Tensor, kappa) -> Tensor:
    t = t.view(-1, 1, 1)
    pt = (1 - kappa(t)) * p0 + kappa(t) * p1
    return sample_p(pt)

def safe_chr(c: int, bos_token: int, pad_token: int, gap_token: int) -> str:
    if c == gap_token: return "<GAP>"
    if c == pad_token: return "<PAD>"
    if c == bos_token: return "<BOS>"
    try:
        ch = chr(c)
        return ch if ch.isprintable() else "."
    except Exception:
        return "."

def pretty_parse(x: torch.Tensor, bos_token: int, pad_token: int, gap_token: int) -> str:
    return ''.join(safe_chr(int(c), bos_token, pad_token, gap_token) for c in x.cpu().tolist())

def _align_pair(a: torch.Tensor, b: torch.Tensor, gap_token: int) -> Tuple[List[int], List[int]]:
    a, b = a.tolist(), b.tolist()
    m, n = len(a), len(b)
    dp = [[i + j if i * j == 0 else 0 for j in range(n + 1)] for i in range(m + 1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1] if a[i-1] == b[j-1] else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    i, j = m, n
    A, B = [], []
    while i or j:
        if i and j and a[i-1] == b[j-1]:
            A.append(a[i-1]); B.append(b[j-1]); i -= 1; j -= 1
        elif i and j and dp[i][j] == dp[i-1][j-1] + 1:
            A.append(a[i-1]); B.append(b[j-1]); i -= 1; j -= 1
        elif i and dp[i][j] == dp[i-1][j] + 1:
            A.append(a[i-1]); B.append(gap_token); i -= 1
        else:
            A.append(gap_token); B.append(b[j-1]); j -= 1
    return A[::-1], B[::-1]

def opt_align_xs_to_zs(x0: torch.Tensor, x1: torch.Tensor, pad_token: int, gap_token: int):
    pairs = [_align_pair(x0[b], x1[b], gap_token) for b in range(x0.size(0))]
    z0 = torch.nn.utils.rnn.pad_sequence([torch.tensor(p[0], device=x0.device) for p in pairs],
                                         batch_first=True, padding_value=pad_token)
    z1 = torch.nn.utils.rnn.pad_sequence([torch.tensor(p[1], device=x1.device) for p in pairs],
                                         batch_first=True, padding_value=pad_token)
    return z0.long(), z1.long()

def rm_gap_tokens(z: torch.Tensor, pad_token: int, gap_token: int):
    batch, seq = z.shape
    filtered = []
    for b in range(batch):
        row = z[b]
        row = row[row != gap_token]
        filtered.append(row[row != pad_token])
    max_len = max(len(r) for r in filtered) if filtered else 0 # need at least one element in xt
    x = torch.full((batch, max_len), pad_token, dtype=z.dtype, device=z.device)
    for b, r in enumerate(filtered):
        x[b, :len(r)] = r
    x_pad = (x == pad_token)
    z_gap = (z == gap_token)
    z_pad = (z == pad_token)
    return x, x_pad, z_gap, z_pad

def rv_gap_tokens(x: torch.Tensor, z_gap: torch.Tensor, z_pad: torch.Tensor, pad_token: int, gap_token: int):
    batch, z_len = z_gap.shape
    out = torch.full((batch, z_len), pad_token, dtype=x.dtype, device=x.device)
    flat = x[x != pad_token]
    ptr = 0
    for b in range(batch):
        write_positions = (~z_gap[b] & ~z_pad[b]).nonzero(as_tuple=True)[0]
        need = write_positions.numel()
        out[b, z_gap[b]] = gap_token
        out[b, write_positions] = flat[ptr:ptr+need]
        ptr += need
    return out

def make_ut_mask_from_z(z_t: torch.Tensor, z_1: torch.Tensor, vocab_size: int, pad_token: int, gap_token: int):
    batch, L = z_t.shape
    n_ops = 2 * vocab_size + 1
    diff = (z_t != z_1) & (z_t != pad_token) & (z_1 != pad_token)
    ins = (z_t == gap_token) & (z_1 != gap_token) & diff
    dele = (z_t != gap_token) & (z_1 == gap_token) & diff
    sub = diff & ~ins & ~dele
    mask = torch.zeros(batch, L, n_ops, dtype=torch.bool, device=z_t.device)
    mask[ins, z_1[ins]] = True
    mask[sub, z_1[sub] + vocab_size] = True
    mask[:, :, -1][dele] = True

    assert diff.sum() == (ins | dele | sub).sum(), "Mismatch in number of edits"
    assert diff.sum() == mask.sum(), 'Mismatch in mask edits'

    return mask


def fill_gap_tokens_with_repeats(ux: torch.Tensor, z_gap: torch.Tensor, z_pad: torch.Tensor):
    b, zL = z_gap.shape
    _, xL, C = ux.shape
    non_gap = ~z_gap
    idx = non_gap.cumsum(dim=1) - 1
    idx = idx.clamp(min=0, max=xL-1)
    batch_idx = torch.arange(b, device=ux.device).unsqueeze(1)
    # out will be of length zt, each entry oi giving the relevant (ins, sub, del) scores for zi
    out = ux[batch_idx, idx]
    out[z_pad] = 0
    return out
