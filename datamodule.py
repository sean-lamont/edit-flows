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
    def __init__(self, tokenizer, batch_size=64, n_train=1000, n_val=100, max_len=7000):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.n_train = n_train
        self.n_val = n_val
        self.max_len = max_len

    def setup(self, stage=None):
        self.ds_train = GoedelDataset()
        self.ds_val = GoedelDataset()

    # for error correction, x0 context is (goal + error + prev_attempt), for inital attempt just goal, target
    def _collate(self, batch):
        x0s = []
        x1s = []
        context_lens = []
        for i in range(len(batch)):
            context = batch[i]['context']
            prev_attempt = batch[i].get('prev_attempt', '')
            target = batch[i]['target']

            context_ids = self.tokenizer(context, return_tensors='pt').input_ids[:self.max_len]
            context_len = context_ids.shape[1]

            prev_ids = self.tokenizer(prev_attempt,  return_tensors='pt').input_ids
            target_ids = self.tokenizer(target,  return_tensors='pt').input_ids

            # combine context and prev_attempt for x0, context and target for x1:
            x0 = torch.cat((context_ids.squeeze(0), prev_ids.squeeze(0)[1:]), dim=0)[:self.max_len]
            x1 = torch.cat((context_ids.squeeze(0), target_ids.squeeze(0)[1:]), dim=0)[:self.max_len]

            x0s.append(x0.unsqueeze(0))
            x1s.append(x1.unsqueeze(0))
            context_lens.append(context_len)

        # todo below assumes that the alignment will keep context at the start for all examples so we can use context_lens
        ret =  collate_batch_goedel(x1s, x0s, pad_token=self.tokenizer.pad_token_id, gap_token=151651)
        ret['context_lens'] = context_lens

        return ret

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self._collate, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size,
                          collate_fn=self._collate, num_workers=0)
