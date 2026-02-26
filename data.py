import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, cast

from flows import Coupling, EmptyCoupling, KappaScheduler, sample_p, x2prob
from utils import opt_align_xs_to_zs, rm_gap_tokens, make_sinusoidal_sequence
from constants import BOS_TOKEN, PAD_TOKEN, GAP_TOKEN


def sample_cond_pt(p0: torch.Tensor, p1: torch.Tensor, t: torch.Tensor, kappa: KappaScheduler):
    t = t.reshape(-1, 1, 1)
    pt = (1 - kappa(t)) * p0 + kappa(t) * p1
    return sample_p(pt)


def make_x0_like_x1(
        x1: torch.Tensor,
        vocab_size: int = 128,
        pad_token: int = PAD_TOKEN,
        noise: float = 0.05,
        **kwargs,
) -> torch.Tensor:
    batch_size, x1_max_len = x1.shape
    x0s = []
    for i in range(batch_size):
        x1i_len = (x1[i] != pad_token).sum().item()
        x0i = torch.Tensor(make_sinusoidal_sequence(int(x1i_len), noise=noise, **kwargs))
        x0i = torch.round(torch.clip(x0i * vocab_size, min=0.0, max=vocab_size - 1)).long()
        x0i = F.pad(x0i, (0, x1_max_len - x0i.shape[0]), value=pad_token)
        x0s.append(x0i)
    x0s = torch.stack(x0s, dim=0).long()  # (batch_size, x1_max_len)
    assert x0s.shape == x1.shape, "x0 and x1 must have the same shape"
    return x0s


def make_x0_with_bounds(
        batch_size: int = 2,
        min_length: int = 96,
        max_length: int = 96,
        vocab_size: int = 128,
        pad_token: int = PAD_TOKEN,
        noise: float = 0.05,
        **kwargs
) -> torch.Tensor:
    lengths = np.random.randint(min_length, max_length + 1, size=(batch_size,))
    max_seq_len = lengths.max()
    x0s = []
    for length in lengths:
        x0i = torch.Tensor(make_sinusoidal_sequence(length, noise=noise, **kwargs))
        x0i = torch.round(torch.clip(x0i * vocab_size, min=0.0, max=vocab_size - 1)).long()
        x0i = F.pad(x0i, (0, max_seq_len - x0i.shape[0]), value=pad_token)
        x0s.append(x0i)
    x0s = torch.stack(x0s, dim=0).long()  # (batch_size, max_seq_len)
    assert x0s.shape[1] == max_seq_len
    assert x0s.shape[0] == batch_size
    return x0s


def make_batch(
        batch_size: int = 2,
        min_length: int = 96,
        max_length: int = 96,
        vocab_size: int = 128,
        pad_token: int = PAD_TOKEN,
        bos_token: int = BOS_TOKEN,
        coupling: Coupling = EmptyCoupling(),
        seq_align_fn=opt_align_xs_to_zs,
        noise: float = 0.05,
        **kwargs,
):
    lengths = np.random.randint(min_length, max_length + 1, size=batch_size)
    x_1, x_0 = [], []
    z_1, z_0 = [], []

    for length in lengths:
        _x1 = torch.Tensor(make_sinusoidal_sequence(length, noise=noise, **kwargs))
        _x1 = torch.round(torch.clip(_x1 * vocab_size, min=0.0, max=vocab_size - 1)).long().unsqueeze(0)
        _x0, _ = coupling.sample(_x1)
        _z0, _z1 = seq_align_fn(_x0, _x1)
        x_1.append(_x1.squeeze(0))
        x_0.append(_x0.squeeze(0))
        z_1.append(_z1.squeeze(0))
        z_0.append(_z0.squeeze(0))

    # Find the maximum length of each sequence in the batch
    x0_max_len = max(len(x) for x in x_0)
    x1_max_len = max(len(x) for x in x_1)
    z_max_len = max(len(z) for z in z_1)
    assert z_max_len == max(len(z) for z in z_0), "z_1 and z_0 must have the same max length"

    # Add <PAD> token at end of each sequence to make them equal length
    x_1 = torch.stack([F.pad(x, (0, x1_max_len - x.shape[0]), value=pad_token) for x in x_1], dim=0).long()
    x_0 = torch.stack([F.pad(x, (0, x0_max_len - x.shape[0]), value=pad_token) for x in x_0], dim=0).long()
    z_1 = torch.stack([F.pad(x, (0, z_max_len - x.shape[0]), value=pad_token) for x in z_1], dim=0).long()
    z_0 = torch.stack([F.pad(x, (0, z_max_len - x.shape[0]), value=pad_token) for x in z_0], dim=0).long()

    # Add <BOS> token at the start of each sequence
    x_1 = F.pad(x_1, (1, 0), value=bos_token)
    x_0 = F.pad(x_0, (1, 0), value=bos_token)
    z_1 = F.pad(z_1, (1, 0), value=bos_token)
    z_0 = F.pad(z_0, (1, 0), value=bos_token)

    t = torch.rand(batch_size, 1)
    t = torch.clamp(t, min=0.01, max=0.99)
    padding_mask = (x_1 == pad_token)
    return x_0, x_1, z_0, z_1, t, padding_mask


def make_ut_mask_from_z(
        z_t: torch.Tensor,
        z_1: torch.Tensor,
        vocab_size: int = 130,
        pad_token: int = PAD_TOKEN,
        gap_token: int = GAP_TOKEN,
) -> torch.Tensor:
    """
    Create a mask for u_cat for indexing the output rate tensor based on differences between z_t and z_1.
    For each position i where z_t and z_1 differ, we index as follows:

    - z_t[i] = GAP_TOKEN & z_1[i] = c => u_mask[i, insert, c] = 1
    - z_t[i] = c & z_1[i] = GAP_TOKEN => u_mask[i, delete] = 1
    - z_t[i] = c1 & z_1[i] = c2 => u_mask[i, substitute, c1, c2] = 1
    """
    batch_size, z_seq_len = z_t.shape
    n_ops = 2 * vocab_size + 1  # insert + substitute + delete

    z_neq = (z_t != z_1) & (z_t != pad_token) & (z_1 != pad_token)
    z_ins = (z_t == gap_token) & (z_1 != gap_token) & z_neq  # (batch_size, z_seq_len)
    z_del = (z_t != gap_token) & (z_1 == gap_token) & z_neq  # (batch_size, z_seq_len)
    z_sub = z_neq & ~z_ins & ~z_del  # (batch_size, z_seq_len)

    # mask (batch_size, z_seq_len, u_ops) where 1 indicates operation that bring z_t closer to z_1
    u_mask = torch.zeros((batch_size, z_seq_len, n_ops), dtype=torch.bool, device=z_t.device)
    u_mask[z_ins, z_1[z_ins]] = True
    u_mask[z_sub, z_1[z_sub] + vocab_size] = True
    u_mask[:, :, -1][z_del] = True

    assert z_neq.sum() == (z_ins | z_del | z_sub).sum(), "Mismatch in number of edits"
    assert z_neq.sum() == u_mask.sum(), "Mismatch in number of edits in mask"

    return u_mask


def fill_gap_tokens_with_repeats(
        x_ut: torch.Tensor,
        z_gap_mask: torch.Tensor,
        z_pad_mask: torch.Tensor,
):
    batch_size, _ = z_gap_mask.shape
    _, x_seq_len, _ = x_ut.shape

    # Use cumsum on non-gap positions to point to the last valid non-gap position
    non_gap_mask = ~z_gap_mask  # Invert mask to get non-gap positions
    indices = non_gap_mask.cumsum(dim=1) - 1  # (batch_size, z_seq_len)
    indices = indices.clamp(min=0, max=x_seq_len - 1)  # Ensure indices are within bounds

    # Use indices to gather from x_ut
    batch_indices = torch.arange(batch_size, device=x_ut.device).unsqueeze(1)
    result = x_ut[batch_indices, indices]  # (batch_size, z_seq_len, vocab_size) (indexing with [b, 1], [b, z_len])
    result[z_pad_mask] = 0  # Set pad positions to 0
    return result