from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
from torch import Tensor
from einops import rearrange


def x2prob(x: Tensor, vocab_size: int) -> Tensor:
    """Converts sequence of tokens to class distribution representation
    """
    return torch.nn.functional.one_hot(x, num_classes=vocab_size).float()


def sample_p(pt: Tensor, temperature: float = 1.0) -> Tensor:
    """Samples protein sequence from class distribution representation
    """
    b, l, _ = pt.shape
    pt = rearrange(pt, 'b l c -> (b l) c')
    xt = torch.multinomial(pt / temperature, 1)
    return xt.reshape(b, l)


class Coupling(ABC):
    @abstractmethod
    def sample(self, x1: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

class SubSequenceCoupling(Coupling):
    def __init__(self, num_subsequences: int, min_size: int, max_size: int, pad_token: int = 129):
        super().__init__()
        self.num_subsequences = num_subsequences
        self.min_size = min_size
        self.max_size = max_size
        self.pad_token = pad_token

    def sample(self, x1: Tensor) -> tuple[Tensor, Tensor]:
        B, L = x1.shape
        device = x1.device

        x0 = torch.full_like(x1, self.pad_token)
        actual_lengths = (x1 != self.pad_token).sum(dim=1)

        for i in range(B):
            seq_len = actual_lengths[i]
            keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)

            if seq_len >= self.min_size:
                for _ in range(self.num_subsequences):
                    current_max_size = min(self.max_size, seq_len)

                    if current_max_size < self.min_size:
                        continue

                    sub_len = torch.randint(self.min_size, current_max_size + 1, (1,))

                    max_start = seq_len - sub_len

                    start_idx = torch.randint(0, max_start + 1, (1,))

                    end_idx = start_idx + sub_len

                    keep_mask[start_idx:end_idx] = False

            valid_tokens = x1[i, :seq_len]
            kept_tokens = valid_tokens[keep_mask]

            x0[i, :len(kept_tokens)] = kept_tokens

        return x1, x0


        return x0, x1


# todo: TBD if sampling will self-correct overshoots (i.e. delete errors).. Could require corrector steps, or
# adding noisy tokens to objective.


class EmptyCoupling(Coupling):
    """A coupling that samples empty prior sequences
    """

    def sample(self, x1: Tensor):
        x0 = torch.empty((x1.shape[0], 0), dtype=x1.dtype, device=x1.device).long()
        return x0, x1


class GeneratorCoupling(Coupling):
    """A coupling that samples prior sequences from a generator function
    """

    def __init__(self, generator_fn: Callable[[Optional[Tensor]], Tensor]):
        self.generator_fn = generator_fn

    def sample(self, x1: Tensor):
        x0 = self.generator_fn(x1)
        return x0, x1


class ExtendedCoupling(Coupling):
    """A coupling that randomly inserts tokens into the target sequence
    """

    def __init__(self, n_insert: int = 10, vocab_size: int = 128, pad_token: int = 129):
        self.n_insert = n_insert
        self.vocab_size = vocab_size
        self.pad_token = pad_token

    def sample(self, x1: Tensor):
        batch_size, x1_seq_len = x1.shape
        x1_pad_mask = (x1 == self.pad_token)
        x1_seq_lengths = (~x1_pad_mask).sum(dim=1).tolist()

        ins_positions = torch.stack([
            torch.randint(0, seqlen + 1, size=(self.n_insert,), dtype=torch.long, device=x1.device)
            for seqlen in x1_seq_lengths
        ])
        ins_positions = torch.sort(ins_positions, dim=1)[0]  # (batch_size, n_insert)

        max_new_len = self.n_insert + x1_seq_len
        x0 = torch.full((batch_size, max_new_len), self.pad_token, dtype=x1.dtype,
                        device=x1.device)  # (batch_size, max_new_len)

        batch_indices = torch.arange(batch_size, device=x1.device).unsqueeze(1)  # (batch_size, 1)
        orig_positions = torch.arange(x1_seq_len, device=x1.device).unsqueeze(0).expand(batch_size,
                                                                                        -1)  # (batch_size, x1_seq_len)

        num_insert_before = (ins_positions.unsqueeze(2) <= orig_positions.unsqueeze(1)).sum(
            dim=1)  # (batch_size, x1_seq_len)
        new_orig_positions = orig_positions + num_insert_before  # (batch_size, x1_seq_len)
        x0[batch_indices, new_orig_positions] = x1

        ins_new_positions = ins_positions + torch.arange(self.n_insert, device=x1.device).unsqueeze(
            0)  # (batch_size, n_insert)
        ins_tokens = torch.randint(0, self.vocab_size, size=(batch_size, self.n_insert), dtype=x1.dtype,
                                   device=x1.device)
        x0[batch_indices, ins_new_positions] = ins_tokens

        return x0, x1


class UniformCoupling(Coupling):
    """A coupling that samples uniform prior sequences within a given length range
    """

    def __init__(
            self,
            min_len: int = 0,
            max_len: int = 100,
            vocab_size: int = 128,
            mirror_len: bool = False,
            pad_token: int = 129,
    ):
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.mirror_len = mirror_len
        self.pad_token = pad_token

    def sample(self, x1: Tensor):
        batch_size, _ = x1.shape
        x1_pad_mask = (x1 == self.pad_token)
        if self.mirror_len:
            x0_pad_mask = x1_pad_mask
            x0_max_len = x1.shape[1]
        else:
            x0_seq_len = torch.randint(self.min_len, self.max_len + 1, size=(batch_size,)).long()
            x0_max_len = int(x0_seq_len.max().item())
            x0_pad_mask = torch.arange(x0_max_len, device=x1.device).expand(batch_size, -1) >= x0_seq_len.unsqueeze(1)

        x0 = torch.randint(0, self.vocab_size, size=(batch_size, x0_max_len), dtype=x1.dtype, device=x1.device)
        x0[x0_pad_mask] = self.pad_token
        return x0, x1


class KappaScheduler(ABC):
    """Base class for kappa schedulers
    """

    @abstractmethod
    def __call__(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def derivative(self, t: Tensor) -> Tensor:
        raise NotImplementedError


class CubicScheduler(KappaScheduler):
    def __init__(self, a: float = 2.0, b: float = 0.5) -> None:
        self.a = a
        self.b = b

    def __call__(self, t: Tensor) -> Tensor:
        return -2 * (t ** 3) + 3 * (t ** 2) + self.a * (t ** 3 - 2 * t ** 2 + t) + self.b * (t ** 3 - t ** 2)

    def derivative(self, t: Tensor) -> Tensor:
        return -6 * (t ** 2) + 6 * t + self.a * (3 * t ** 2 - 4 * t + 1) + self.b * (3 * t ** 2 - 2 * t)
