from typing import Optional, Dict, Any
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from transformers import AutoTokenizer
from goedel_dataset import GoedelDataset
import torch
import torch.nn.functional as F
import os

from torch.nn.utils.rnn import pad_sequence
from scheduler import CubicScheduler

try:
    from utils import make_ut_mask_from_z
except ImportError:
    print("Error: Could not import 'make_ut_mask_from_z' from 'utils.py'.")
    print("Please make sure 'datamodule.py' is in the same directory as 'utils.py'.")
    exit(1)


class AdaptedDataModule(pl.LightningDataModule):
    def __init__(self,
                 tokenizer: str,
                 full_vocab_size: int,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 data_path: str = "precomputed_hf_dataset",
                 scheduler_cfg: Dict[str, Any] | None = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.data_path = data_path
        self.full_vocab_size = full_vocab_size

        self.num_workers = num_workers if num_workers is not None else os.cpu_count()

        self.kappa = CubicScheduler(**(scheduler_cfg or {'a': 0.0, 'b': 2.0}))

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.pad_token = self.tokenizer.pad_token_id
        self.gap_token = 151651

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_dataset = GoedelDataset(folder_path=self.data_path)

            split_dataset = full_dataset.dataset.train_test_split(
                test_size=0.05, seed=42
            )

            self.ds_train = split_dataset['train']
            self.ds_val = split_dataset['test']

            self.ds_train.set_format(type='torch')
            self.ds_val.set_format(type='torch')

    def _collate(self, batch):

        # 1. Unzip the batch
        x0s = [item['x0'] for item in batch]
        z0s = [item['z0'] for item in batch]
        z1s = [item['z1'] for item in batch]
        contexts = [item['context'] for item in batch]
        x1s = [item['x1'] for item in batch]
        idx = [item.get('idx', -1) for item in batch]
        type = [item.get('type', 'unknown') for item in batch]

        # 2. Pad z0 and z1 efficiently
        z0 = pad_sequence(z0s, batch_first=True, padding_value=self.pad_token).long()
        z1 = pad_sequence(z1s, batch_first=True, padding_value=self.pad_token).long()

        # 3. Sample t
        t = torch.rand(len(batch), 1)
        t = torch.clamp(t - 1e-2, min=0.0)

        # 4. Calculate zt (stochastic interpolation)
        k_t = self.kappa(t)
        k_t_broadcast = k_t.view(-1, 1).expand_as(z0)

        rand_probs = torch.rand_like(z0, dtype=torch.float32)
        zt = torch.where(rand_probs < k_t_broadcast, z1, z0)

        # 5. --- MODIFIED: Generate z_pad and z_gap from zt ---
        # This now correctly follows the original logic
        z_pad = (zt == self.pad_token)
        z_gap = (zt == self.gap_token)

        # 6. Calculate uz_mask
        uz_mask = make_ut_mask_from_z(
            zt, z1, self.full_vocab_size, self.pad_token, self.gap_token
        )

        # 7. Remove gap tokens to create xt (variable length)
        xts = []
        for i in range(len(batch)):
            zt_sample = zt[i]
            # We can use the pre-computed z_gap mask here, but this is clearer
            xt_sample = zt_sample[zt_sample != self.gap_token]
            xts.append(xt_sample)

        # 8. Pad xt, context, x0 and x1 efficiently
        xt = pad_sequence(xts, batch_first=True, padding_value=self.pad_token).long()
        xt_pad = (xt == self.pad_token)

        contexts_padded = pad_sequence(contexts, batch_first=True, padding_value=self.pad_token).long()
        x1_padded = pad_sequence(x1s, batch_first=True, padding_value=self.pad_token).long()
        x0_padded = pad_sequence(x0s, batch_first=True, padding_value=self.pad_token).long()

        # 9. Return the batch, ready for the GPU
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
        return DataLoader(self.ds_train, batch_size=self.batch_size,
                          shuffle=True,
                          collate_fn=self._collate, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size,
                          collate_fn=self._collate, num_workers=self.num_workers, shuffle=False)