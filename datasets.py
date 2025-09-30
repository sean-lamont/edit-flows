import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Callable
from utils import *

def make_sinusoidal_sequence(
    length: int,
    noise: float,
    num_cycles_fn: Callable[[], float],
    x_int_fn: Callable[[], float]
):
    x = np.linspace(0, 4*np.pi, length)
    B = 2 * np.pi * num_cycles_fn() / (4*np.pi)
    C = x_int_fn()
    y = 0.5 * np.sin(B * (x - C)) + 0.5
    if noise > 0:
        y += np.random.normal(0, noise, length)
    y = np.clip(y * BASE_VOCAB, 0, BASE_VOCAB-1)
    return torch.round(torch.tensor(y)).long().unsqueeze(0)

class SinusoidDataset(Dataset):
    def __init__(self, n_samples=10000, min_len=64, max_len=128, noise=0.05,
                 num_cycles_fn=lambda: np.random.uniform(1.5, 4.0),
                 x_int_fn=lambda: np.random.uniform(0, 2*np.pi)):
        self.n = n_samples
        self.min = min_len
        self.max = max_len
        self.noise = noise
        self.num_cycles_fn = num_cycles_fn
        self.x_int_fn = x_int_fn
    def __len__(self):
        return self.n
    def __getitem__(self, idx: int):
        L = np.random.randint(self.min, self.max+1)
        seq = make_sinusoidal_sequence(L, self.noise, self.num_cycles_fn, self.x_int_fn)
        return seq