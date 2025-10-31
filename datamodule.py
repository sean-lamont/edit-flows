from typing import Optional
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from collate import collate_batch, collate_batch_goedel
from couplings import Coupling, EmptyCoupling
from transformers import AutoTokenizer
from datasets import GoedelDataset
import torch

class AdaptedDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, batch_size=64, n_train=1000, n_val=100, max_context_len=2000, max_len=4000):
        super().__init__()
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.n_train = n_train
        self.n_val = n_val
        self.max_context_len = max_context_len
        self.max_len = max_len

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            # 1. Load your full dataset
            self.full_dataset = GoedelDataset()

            # 2. Calculate split lengths
            dataset_size = len(self.full_dataset)
            train_size = int(dataset_size * 0.95)
            val_size = dataset_size - train_size  # Ensure it sums up correctly

            # 3. Perform the random split
            # Use a generator for reproducibility
            generator = torch.Generator().manual_seed(42)
            self.ds_train, self.ds_val = random_split(
                self.full_dataset,
                [train_size, val_size],
                generator=generator
            )

    def _collate(self, batch):
        x0s = []
        x1s = []
        context_lens = []
        contexts = []
        for i in range(len(batch)):
            context = batch[i]['context']
            prev_attempt = batch[i].get('prev_attempt', 'INITIAL ATTEMPT') # set to nonzero beginning
            target = batch[i]['target']

            # print (f'context: {context}\n prev: {prev_attempt} \n target: {target}')

            context_ids = self.tokenizer(context, return_tensors='pt').input_ids[:self.max_context_len]
            context_len = context_ids.shape[1]

            prev_ids = self.tokenizer(prev_attempt,  return_tensors='pt').input_ids[:self.max_len]
            target_ids = self.tokenizer(target,  return_tensors='pt').input_ids[:self.max_len]

            x0s.append(prev_ids)
            x1s.append(target_ids)
            context_lens.append(context_len)
            contexts.append(context_ids.squeeze(0))

        ret =  collate_batch_goedel(x1s, x0s, pad_token=self.tokenizer.pad_token_id, gap_token=151651)
        ret['context_lens'] = context_lens
        ret['contexts'] = contexts
        ret['idx'] = [b['idx'] for b in batch]
        ret['type'] = [b['type'] for b in batch]

        return ret

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size,
                          shuffle=False,
                          collate_fn=self._collate, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size,
                          collate_fn=self._collate, num_workers=0, shuffle=False)
