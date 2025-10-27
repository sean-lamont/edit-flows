from typing import List

import torch
import torch.nn.functional as F

from couplings import Coupling, EmptyCoupling
from utils import opt_align_xs_to_zs


# for proof data, x0 and  x1 are given
def collate_batch_goedel(x1_list, x0_list, pad_token: int, gap_token: int):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # x1_list = [b.to(device) for b in x1s]
    # x0_list = [b.to(device) for b in x0s]

    assert len(x0_list) == len(x1_list)


    x_1, x_0 = [], []
    z_1, z_0 = [], []
    for i in range(len(x1_list)):
        _x1 = x1_list[i]
        # _x0, _ = coupling.sample(_x1)
        _x0 = x0_list[i]
        _z0, _z1 = opt_align_xs_to_zs(_x0, _x1, pad_token, gap_token)
        x_1.append(_x1.squeeze(0))
        x_0.append(_x0.squeeze(0))
        z_1.append(_z1.squeeze(0))
        z_0.append(_z0.squeeze(0))

    x0_max_len = max([len(x) for x in x_0]) if x_0 else 0
    x1_max_len = max([len(x) for x in x_1]) if x_1 else 0
    z_max_len = max([len(z) for z in z_1]) if z_1 else 0

    if z_0:
        assert z_max_len == max(len(z) for z in z_0), "z_1 and z_0 must have the same max length"

    # add <pad> token at end of each sequence to make them equal length
    x_1 = torch.stack([F.pad(x, (0, x1_max_len - x.shape[0]), value=pad_token) for x in x_1], dim=0).long()
    x_0 = torch.stack([F.pad(x, (0, x0_max_len - x.shape[0]), value=pad_token) for x in x_0], dim=0).long()
    z_1 = torch.stack([F.pad(x, (0, z_max_len - x.shape[0]), value=pad_token) for x in z_1], dim=0).long()
    z_0 = torch.stack([F.pad(x, (0, z_max_len - x.shape[0]), value=pad_token) for x in z_0], dim=0).long()

    t = torch.rand(len(x1_list), 1) # subtract eps to account for occasional 1's

    if torch.any(t == 1.0):
        print ('time=1' + '!' * 50)
        t = torch.clamp(t - 1e-2, min=0.0)

    padding_mask = (x_1 == pad_token)

    return {
        'x0': x_0.long(), 'x1': x_1.long(),
        'z0': z_0.long(), 'z1': z_1.long(),
        't': t, 'pad_mask': padding_mask,
    }

# given batch of x1s and coupling distribution, get x0s and alignments zs, pad and add BOS tokens
def collate_batch(batch: List[torch.Tensor],  bos_token: int, pad_token: int, gap_token: int, coupling: Coupling | None = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coupling = coupling or EmptyCoupling()
    x1_list = [b.to(device) for b in batch]

    x_1, x_0 = [], []
    z_1, z_0 = [], []
    for _x1 in x1_list:
        _x0, _ = coupling.sample(_x1)
        _z0, _z1 = opt_align_xs_to_zs(_x0, _x1, pad_token, gap_token)
        x_1.append(_x1.squeeze(0))
        x_0.append(_x0.squeeze(0))
        z_1.append(_z1.squeeze(0))
        z_0.append(_z0.squeeze(0))

    x0_max_len = max([len(x) for x in x_0]) if x_0 else 0
    x1_max_len = max([len(x) for x in x_1]) if x_1 else 0
    z_max_len = max([len(z) for z in z_1]) if z_1 else 0

    if z_0:
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

    t = torch.rand(len(x1_list), 1)
    padding_mask = (x_1 == pad_token)

    return {
        'x0': x_0.long(), 'x1': x_1.long(),
        'z0': z_0.long(), 'z1': z_1.long(),
        't': t, 'pad_mask': padding_mask,
    }
