"""

This module implements the coupling from source to target distributions.
Currently not used in the main codebase, as we have x0 and x1 given directly.


"""


from abc import ABC, abstractmethod
from typing import Optional, Callable
import torch
from torch import Tensor
from utils import *

class Coupling(ABC):
    @abstractmethod
    def sample(self, x1: Tensor) -> tuple[Tensor, Tensor]:
        ...

class EmptyCoupling(Coupling):
    def sample(self, x1: Tensor):
        x0 = torch.empty((x1.size(0), 0), dtype=x1.dtype, device=x1.device).long()
        return x0, x1

class GeneratorCoupling(Coupling):
    def __init__(self, fn: Callable[[Tensor], Tensor]):
        self.fn = fn
    def sample(self, x1: Tensor):
        return self.fn(x1), x1

class UniformCoupling(Coupling):
    def __init__(self, min_len=0, max_len=100, vocab=1, pad=0):
        self.min_len = min_len; self.max_len = max_len; self.vocab = vocab; self.pad = pad
    def sample(self, x1: Tensor):
        b = x1.size(0)
        lens = torch.randint(self.min_len, self.max_len+1, (b,), device=x1.device)
        L = lens.max().item()
        x0 = torch.randint(0, self.vocab, (b, L), device=x1.device)
        mask = torch.arange(L, device=x1.device).unsqueeze(0) >= lens.unsqueeze(1)
        x0[mask] = self.pad
        return x0, x1