"""
This module defines the PyTorch Lightning DataModule for the project.

It encapsulates all data-related steps, including loading datasets from files,
splitting them into training, validation, and test sets, and creating PyTorch
DataLoaders for each.

It expects a preprocessed dataset, where each batch contains the precomputed aligned
sequences 'z0' and 'z1', as well as the original sequences 'x0' and 'x1', and the context.


"""


from typing import Optional, Dict, Any
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
import os
import datasets

from torch.nn.utils.rnn import pad_sequence
from scheduler import CubicScheduler

from utils import make_ut_mask_from_z, opt_align_xs_to_zs
from dataset.sinusoidal import SinusoidalDataset, make_sinusoidal_sequence


class AdaptedDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset,
                 tokenizer: str,
                 full_vocab_size: int,
                 gap_token,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 scheduler_cfg: Dict[str, Any] | None = None,
                 dataset_cfg: Dict[str, Any] | None = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.full_vocab_size = full_vocab_size

        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        self.kappa = CubicScheduler(**(scheduler_cfg or {'a': 0.0, 'b': 2.0}))

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_token = self.tokenizer.pad_token_id
        self.gap_token = gap_token
        self.dataset_name = dataset
        self.dataset_cfg = dataset_cfg or {}

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.dataset_name == "sinusoidal":
                self.ds_train = SinusoidalDataset(num_samples=10000, **self.dataset_cfg)
                self.ds_val = SinusoidalDataset(num_samples=1000, **self.dataset_cfg)
            else:
                self.dataset = datasets.load_dataset(self.dataset_name)
                self.ds_train = self.dataset['train']
                self.ds_val = self.dataset['test']

                self.ds_train.set_format(type='torch')
                self.ds_val.set_format(type='torch')

    def _collate_sinusoidal(self, batch):
        x1s = batch
        x0s = []
        for x1 in x1s:
            x0_np = make_sinusoidal_sequence(len(x1), noise=0.05, num_cycles_fn=lambda: 1, x_int_fn=lambda: 0)
            x0 = torch.tensor(np.clip(self.dataset_cfg.get("vocab_size", 128) * x0_np, 0, self.dataset_cfg.get("vocab_size", 128) - 1)).long()
            x0s.append(x0)

        z0s, z1s = [], []
        for x0, x1 in zip(x0s, x1s):
            z0, z1 = opt_align_xs_to_zs(x0.unsqueeze(0), x1.unsqueeze(0))
            z0s.append(z0.squeeze(0))
            z1s.append(z1.squeeze(0))

        z0 = pad_sequence(z0s, batch_first=True, padding_value=self.pad_token).long()
        z1 = pad_sequence(z1s, batch_first=True, padding_value=self.pad_token).long()

        t = torch.rand(len(batch), 1)
        t = torch.clamp(t - 1e-2, min=0.0)

        k_t = self.kappa(t)
        k_t_broadcast = k_t.view(-1, 1).expand_as(z0)

        rand_probs = torch.rand_like(z0, dtype=torch.float32)
        zt = torch.where(rand_probs < k_t_broadcast, z1, z0)

        z_pad = (zt == self.pad_token)
        z_gap = (zt == self.gap_token)

        uz_mask = make_ut_mask_from_z(
            zt, z1, self.full_vocab_size, self.pad_token, self.gap_token
        )

        xts = []
        for i in range(len(batch)):
            zt_sample = zt[i]
            xt_sample = zt_sample[zt_sample != self.gap_token]
            xts.append(xt_sample)

        xt = pad_sequence(xts, batch_first=True, padding_value=self.pad_token).long()
        xt_pad = (xt == self.pad_token)

        x1_padded = pad_sequence(x1s, batch_first=True, padding_value=self.pad_token).long()
        x0_padded = pad_sequence(x0s, batch_first=True, padding_value=self.pad_token).long()

        return {
            'xt': xt,
            'contexts': None,
            't': t,
            'x1_padded': x1_padded,
            'x0_padded': x0_padded,
            'uz_mask': uz_mask,
            'xt_pad': xt_pad,
            'z_gap': z_gap,
            'z_pad': z_pad,
            'idx': [-1] * len(batch),
            'type': ['sinusoidal'] * len(batch)
        }


    def _collate_hf(self, batch):
        x0s = [item['x0'] for item in batch]
        z0s = [item['z0'] for item in batch]
        z1s = [item['z1'] for item in batch]
        contexts = [item['context'] for item in batch]
        x1s = [item['x1'] for item in batch]
        idx = [item.get('idx', -1) for item in batch]
        type = [item.get('type', 'unknown') for item in batch]

        z0 = pad_sequence(z0s, batch_first=True, padding_value=self.pad_token).long()
        z1 = pad_sequence(z1s, batch_first=True, padding_value=self.pad_token).long()

        t = torch.rand(len(batch), 1)
        t = torch.clamp(t - 1e-2, min=0.0)

        k_t = self.kappa(t)
        k_t_broadcast = k_t.view(-1, 1).expand_as(z0)

        rand_probs = torch.rand_like(z0, dtype=torch.float32)
        zt = torch.where(rand_probs < k_t_broadcast, z1, z0)

        z_pad = (zt == self.pad_token)
        z_gap = (zt == self.gap_token)

        uz_mask = make_ut_mask_from_z(
            zt, z1, self.full_vocab_size, self.pad_token, self.gap_token
        )

        xts = []
        for i in range(len(batch)):
            zt_sample = zt[i]
            xt_sample = zt_sample[zt_sample != self.gap_token]
            xts.append(xt_sample)

        xt = pad_sequence(xts, batch_first=True, padding_value=self.pad_token).long()
        xt_pad = (xt == self.pad_token)

        contexts_padded = pad_sequence(contexts, batch_first=True, padding_value=self.pad_token).long()
        x1_padded = pad_sequence(x1s, batch_first=True, padding_value=self.pad_token).long()
        x0_padded = pad_sequence(x0s, batch_first=True, padding_value=self.pad_token).long()

        return {
            'xt': xt,
            'contexts': contexts_padded,
            't': t,
            'x1_padded': x1_padded,
            'x0_padded': x0_padded,
            'uz_mask': uz_mask,
            'xt_pad': xt_pad,
            'z_gap': z_gap,
            'z_pad': z_pad,
            'idx': idx,
            'type': type
        }

    def train_dataloader(self):
        collate_fn = self._collate_sinusoidal if self.dataset_name == "sinusoidal" else self._collate_hf
        return DataLoader(self.ds_train, batch_size=self.batch_size,
                          # shuffle=True,
                          shuffle=False,
                          collate_fn=collate_fn, num_workers=self.num_workers)

    def val_dataloader(self):
        collate_fn = self._collate_sinusoidal if self.dataset_name == "sinusoidal" else self._collate_hf
        return DataLoader(self.ds_val, batch_size=self.batch_size,
                          collate_fn=collate_fn, num_workers=self.num_workers, shuffle=False)