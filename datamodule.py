from typing import Optional
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from datasets import SinusoidDataset, GoedelDataset
from collate import collate_batch, collate_batch_goedel
from couplings import Coupling, EmptyCoupling
from transformers import AutoTokenizer
import torch

class SinusoidDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, n_train=5000, n_val=512,
                 min_len=64, max_len=128, coupling: Optional[Coupling] = None):
        super().__init__()
        self.batch_size = batch_size
        self.n_train = n_train
        self.n_val = n_val
        self.min_len = min_len
        self.max_len = max_len
        self.coupling = coupling or EmptyCoupling()

    def setup(self, stage=None):
        self.ds_train = SinusoidDataset(n_samples=self.n_train, min_len=self.min_len, max_len=self.max_len)
        self.ds_val = SinusoidDataset(n_samples=self.n_val, min_len=self.min_len, max_len=self.max_len)

    def _collate(self, batch):
        return collate_batch(batch, coupling=self.coupling)

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self._collate, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size,
                          collate_fn=self._collate, num_workers=0)

class AdaptedDataModule(pl.LightningDataModule):
    def __init__(self, pretrained_model_name: str, batch_size=64, n_train=1000, n_val=100, coupling: Optional[Coupling] = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.batch_size = batch_size
        self.n_train = n_train
        self.n_val = n_val
        self.coupling = coupling or EmptyCoupling()

    def setup(self, stage=None):
        self.ds_train = GoedelDataset(n_samples=self.n_train)
        self.ds_val = GoedelDataset(n_samples=self.n_val)

    # todo for error correction, x0 should be context with old response, x1 should be context with new response
    def _collate(self, batch):
        contexts = [item["context"] for item in batch]
        responses = [item["response"] for item in batch]

        # For now, we'll just use the response as x1 and an empty sequence as x0
        x1s = self.tokenizer(responses, padding='longest', truncation=True, return_tensors='pt')['input_ids']
        x0s = self.tokenizer(contexts, padding='longest', truncation=True, return_tensors='pt')['input_ids']

        # for initial attempt, x0 is just the context and x1 is the response.

        return collate_batch_goedel(x1s, x0s)
        
    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self._collate, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size,
                          collate_fn=self._collate, num_workers=0)
