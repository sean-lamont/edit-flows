import torch
import torch.nn.functional as F
from typing import List, Tuple

BOS_TOKEN = 128
PAD_TOKEN = 129
GAP_TOKEN = 130


def _align_pair(seq_0: torch.Tensor, seq_1: torch.Tensor) -> Tuple[List[int], List[int]]:
    """
    Aligns two sequences using dynamic programming to find the minimum edit distance.
    Returns two lists representing the aligned sequences.
    """
    seq_0, seq_1 = seq_0.cpu().numpy(), seq_1.cpu().numpy()
    m, n = len(seq_0), len(seq_1)
    
    # DP table
    dp = [[i + j if i == 0 or j == 0 else 0 for j in range(n + 1)] for i in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] if seq_0[i-1] == seq_1[j-1] else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    # Backtrack
    aligned_0, aligned_1 = [], []
    i, j = m, n
    while i or j:
        if i and j and seq_0[i-1] == seq_1[j-1]:
            aligned_0.append(seq_0[i-1])
            aligned_1.append(seq_1[j-1])
            i, j = i-1, j-1
        elif i and j and dp[i][j] == dp[i-1][j-1] + 1:
            aligned_0.append(seq_0[i-1])
            aligned_1.append(seq_1[j-1])
            i, j = i-1, j-1
        elif i and dp[i][j] == dp[i-1][j] + 1:
            aligned_0.append(seq_0[i-1])
            aligned_1.append(GAP_TOKEN)
            i -= 1
        else:
            aligned_0.append(GAP_TOKEN)
            aligned_1.append(seq_1[j-1])
            j -= 1
    
    return aligned_0[::-1], aligned_1[::-1]


def naive_align_xs_to_zs(x_0: torch.Tensor, x_1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aligns x_0 and x_1 to the same length by padding with gap_token.
    """
    max_len = max(x_0.shape[1], x_1.shape[1])
    x_0_padded = F.pad(x_0, (0, max_len - x_0.shape[1]), value=GAP_TOKEN)
    x_1_padded = F.pad(x_1, (0, max_len - x_1.shape[1]), value=GAP_TOKEN)
    return x_0_padded, x_1_padded


def shifted_align_xs_to_zs(x_0: torch.Tensor, x_1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aligns x_0 and z_1 by shifting x_1 to the right by the length of x_0, then
    padding all sequences to the same length with gap tokens.
    """
    batch_size, _ = x_0.shape
    x0_seq_lens = (~(x_0 == GAP_TOKEN)).sum(dim=1)
    x1_seq_lens = (~(x_1 == GAP_TOKEN)).sum(dim=1)
    z_seq_lens = x0_seq_lens + x1_seq_lens
    max_z_len = int(z_seq_lens.max().item())
    z_0 = torch.full((batch_size, max_z_len), GAP_TOKEN, dtype=x_0.dtype, device=x_0.device)
    z_1 = torch.full((batch_size, max_z_len), GAP_TOKEN, dtype=x_1.dtype, device=x_1.device)
    batch_indices = torch.arange(batch_size, device=x_0.device).unsqueeze(1)
    z_0[batch_indices, :x0_seq_lens] = x_0
    z_1[batch_indices, x0_seq_lens:] = x_1
    z_0[batch_indices, z_seq_lens:] = PAD_TOKEN
    z_1[batch_indices, z_seq_lens:] = PAD_TOKEN
    return z_0, z_1


def opt_align_xs_to_zs(x_0: torch.Tensor, x_1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aligns x_0 and x_1 to the same length by using a dynamic programming approach
    to find the minimum edit distance alignment.
    """
    aligned_pairs = [_align_pair(x_0[b], x_1[b]) for b in range(x_0.shape[0])]
    x_0_aligned = torch.stack(
        [torch.tensor(pair[0], dtype=x_0.dtype, device=x_0.device) for pair in aligned_pairs])
    x_1_aligned = torch.stack(
        [torch.tensor(pair[1], dtype=x_1.dtype, device=x_1.device) for pair in aligned_pairs])
    return x_0_aligned, x_1_aligned


def rm_gap_tokens(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Remove gap tokens from a batched tensor and right-pad with PAD_TOKEN.
    """    
    batch_size, _ = z.shape
    z_no_gap = []
    for b in range(batch_size):
        z_no_pad = z[b][z[b] != PAD_TOKEN]
        z_no_gap.append(z_no_pad[z_no_pad != GAP_TOKEN])
    max_len = max(len(z) for z in z_no_gap)
    x = torch.stack([F.pad(z, (0, max_len - len(z)), value=PAD_TOKEN) for z in z_no_gap], dim=0).long()
    x_pad_mask = (x == PAD_TOKEN)
    z_gap_mask = (z == GAP_TOKEN)
    z_pad_mask = (z == PAD_TOKEN)
    assert ((~x_pad_mask).sum(1) + z_gap_mask.sum(1)).equal((~z_pad_mask).sum(1))
    return x, x_pad_mask, z_gap_mask, z_pad_mask


def rv_gap_tokens(x: torch.Tensor, z_gap_mask: torch.Tensor, z_pad_mask: torch.Tensor) -> torch.Tensor:
    """
    Reinsert gap tokens into a tensor at specified positions.
    """
    assert x.shape[0] == z_gap_mask.shape[0]
    assert x.shape[1] <= z_gap_mask.shape[1]
    assert z_gap_mask.shape == z_pad_mask.shape
    batch_size, _ = x.shape
    _, z_seq_len = z_gap_mask.shape
    z = torch.full((batch_size, z_seq_len), PAD_TOKEN, dtype=x.dtype, device=x.device)    
    z[~z_gap_mask & ~z_pad_mask] = x[x != PAD_TOKEN]
    z[z_gap_mask] = GAP_TOKEN
    return z


def safe_chr(c: int, show_special_chars=False, compact=False) -> str:
    if c == GAP_TOKEN:
        return 'Δ' if compact else '<GAP>'
    elif c == PAD_TOKEN:
        return 'π' if compact else '<PAD>'
    elif c == BOS_TOKEN:
        return '<BOS>'
    try:
        ch = chr(c)
        # Replace non-printable or whitespace (except space) with '.'
        if ch.isprintable() and (ch == ' ' or not ch.isspace()):
            return ch
        elif show_special_chars:
            return repr(ch)
        else:
            return '.'
    except Exception:
        return '.'


def pretty_parse(x: torch.Tensor, **kwargs) -> str:
    x_str = ''.join(safe_chr(int(c), **kwargs) for c in x.cpu().numpy().flatten())
    return x_str


def pretty_print(x: torch.Tensor, **kwargs) -> None:
    """
    Pretty print a tensor as an ascii string with gap tokens represented as '-'
    Non-printable/special characters (including line breaks, tabs, etc.) are replaced with '.'
    """
    print(pretty_parse(x, **kwargs))


if __name__ == "__main__":
    
    z = torch.tensor([
        [1, 2, 3, GAP_TOKEN, 5, 6, 7],
        [6, 7, GAP_TOKEN, GAP_TOKEN, 10, PAD_TOKEN, PAD_TOKEN],
        [11, 12, 13, 14, 15, 16, PAD_TOKEN],
    ])
    
    x, x_pad_mask, z_gap_mask, z_pad_mask = rm_gap_tokens(z)
    zr = rv_gap_tokens(x, z_gap_mask, z_pad_mask)
    
    print(f"Original Tensor: {z.shape}\n", z, "\n")
    print(f"Padded Tensor: {x.shape}\n", x, "\n")
    print(f"Padding Mask: {x_pad_mask.shape}\n", x_pad_mask, "\n")
    print(f"Gap Mask: {z_gap_mask.shape}\n", z_gap_mask, "\n")
    print(f"Reconstructed Tensor: {zr.shape}\n", zr, "\n")
    assert torch.equal(z, zr), "Reconstructed tensor does not match the original tensor."
    print("Reconstruction successful!")

    