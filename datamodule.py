from typing import Optional
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from datasets import SinusoidDataset
from collate import collate_batch
from couplings import Coupling, EmptyCoupling

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