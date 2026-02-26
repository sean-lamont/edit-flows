import torch
import torch.nn.functional as F
from typing import List, Tuple, Callable, cast
import matplotlib.pyplot as plt
import numpy as np

from constants import BOS_TOKEN, PAD_TOKEN, GAP_TOKEN
from flows import KappaScheduler
from model import SimpleEditFlowsTransformer


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
    batch_size, z_len = z.shape
    device = z.device

    z_gap_mask = (z == GAP_TOKEN)
    z_pad_mask = (z == PAD_TOKEN)
    
    # Mask for tokens to keep (neither GAP nor PAD)
    keep_mask = ~z_gap_mask & ~z_pad_mask
    
    # Get the values and their original batch indices
    kept_values = z[keep_mask]
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, z_len)[keep_mask]

    # Calculate new positions and scatter into a new tensor
    new_lengths = keep_mask.sum(dim=1)
    max_len = new_lengths.max().item()
    x = torch.full((batch_size, max_len), PAD_TOKEN, dtype=z.dtype, device=device)
    new_pos = torch.arange(z_len, device=device).unsqueeze(0).expand(batch_size, -1)[keep_mask] - z_gap_mask.cumsum(dim=1)[keep_mask]
    x[batch_indices, new_pos] = kept_values

    x_pad_mask = (x == PAD_TOKEN)
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


def make_sinusoidal_sequence(
        x_seq_len: int,
        noise: float,
        num_cycles_fn: Callable[[], float] = lambda: np.random.uniform(1.5, 3.5),
        x_int_fn: Callable[[], float] = lambda: np.random.uniform(0, 2 * np.pi)
) -> np.ndarray:
    """
    Generate a discretized sinusoidal sequence with optional Gaussian noise.
    The sinusoidal function follows: y = 1/2 * sin(B(x-C)) + 1/2 where B and C are randomly chosen.
    """
    x = np.linspace(0, 4 * np.pi, x_seq_len)
    num_cycles = num_cycles_fn()
    B = 2 * np.pi * num_cycles / (4 * np.pi)
    C = x_int_fn()
    y = 0.5 * np.sin(B * (x - C)) + 0.5
    if noise > 0:
        gaussian_noise = np.random.normal(0, noise, x_seq_len)
        y += gaussian_noise
    return y


def plot_sequences(xs: np.ndarray, title: str = "Sequences", pad_token: int | None = None):
    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.grid()
    colors = plt.cm.viridis(np.linspace(0, 1, xs.shape[0]))
    for i, y in enumerate(xs):
        if pad_token is not None:
            y = y[y != pad_token]
        x = np.arange(len(y))
        plt.scatter(x, y, label=f"Sequence {i + 1}", s=10, c=colors[i])
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def apply_ins_del_operations(
        x_t: torch.Tensor,
        ins_mask: torch.Tensor,
        del_mask: torch.Tensor,
        ins_tokens: torch.Tensor,
        max_seq_len: int = 512,
        pad_token=GAP_TOKEN,
) -> torch.Tensor:
    """
    Apply insertion and deletion operations to a sequence x_t based on the provided masks.
    """
    batch_size, seq_len = x_t.shape
    device = x_t.device

    # Handle simultaneous ins+del as substitutions
    replace_mask = ins_mask & del_mask
    x_t_modified = x_t.clone()
    x_t_modified[replace_mask] = ins_tokens[replace_mask]

    # Update ins/del masks after handling replacements
    eff_ins_mask = ins_mask & ~replace_mask
    eff_del_mask = del_mask & ~replace_mask

    # Compute new lengths after applying ins/del operations
    xt_pad_mask = (x_t == pad_token)  # (batch_size, seq_len)
    xt_seq_lens = (~xt_pad_mask).sum(dim=1)  # (batch_size,)
    new_lengths = xt_seq_lens + eff_ins_mask.sum(dim=1) - eff_del_mask.sum(dim=1)
    max_new_len = int(new_lengths.max().item())

    if max_new_len <= 0:
        print(f"Unexpected max_new_len <= 0: {max_new_len}, did we delete everything?")
        return torch.full((batch_size, 1), pad_token, dtype=x_t.dtype, device=device)

    # Pre-allocate result
    x_new = torch.full((batch_size, max_new_len), pad_token, dtype=x_t.dtype, device=device)

    # Compute positions
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1)  # (batch_size, 1)
    pos_idx = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
    cum_del = torch.cumsum(eff_del_mask.float(), dim=1)  # num del up to & incl. current pos
    cum_ins = torch.cumsum(eff_ins_mask.float(), dim=1)  # num ins up to & incl. current pos
    cum_ins_before = F.pad(cum_ins[:, :-1], (1, 0), value=0)  # num ins before current pos

    # Place non-deleted tokens
    new_pos = pos_idx + cum_ins_before - cum_del  # new pos of tokens shifted by ins/del
    keep_mask = ~eff_del_mask & (new_pos >= 0) & (new_pos < max_new_len)  # tokens to keep (non-deleted)
    if keep_mask.any():
        x_new[batch_idx.expand(-1, seq_len)[keep_mask], new_pos[keep_mask].long()] = x_t_modified[keep_mask]

    # Place insertions
    if eff_ins_mask.any():
        ins_pos = new_pos + 1  # insertions go 1 after new shifted pos
        ins_valid = eff_ins_mask & (ins_pos >= 0) & (ins_pos < max_new_len)  # tokens to insert
        if ins_valid.any():
            x_new[batch_idx.expand(-1, seq_len)[ins_valid], ins_pos[ins_valid].long()] = ins_tokens[ins_valid]

    if max_new_len > max_seq_len:
        print(f"Warning: max_new_len {max_new_len} exceeds max_seq_len {max_seq_len}, truncating.")
        max_new_len = max_seq_len

    return x_new[:, :max_new_len]


def get_adaptive_h(h: float, t: torch.Tensor, scheduler: KappaScheduler):
    coeff = (1 - scheduler(t)) / scheduler.derivative(t)
    _h = h * torch.ones_like(t, device=t.device)
    h_adapt = torch.minimum(_h, coeff)
    return h_adapt


def load_model_state(model_name: str, device: torch.device):
    checkpoint = torch.load(model_name, map_location=device)
    model = SimpleEditFlowsTransformer(
        vocab_size=checkpoint['vocab_size'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        num_heads=checkpoint['num_heads'],
        max_seq_len=checkpoint['max_seq_len'],
        bos_token_id=checkpoint['bos_token_id'],
        pad_token_id=checkpoint['pad_token_id'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    # optim.load_state_dict(checkpoint['optimizer_state_dict'])
    return model.to(device)#, optim