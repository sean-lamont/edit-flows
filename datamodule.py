from typing import Optional
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from collate import collate_batch, collate_batch_goedel
from couplings import Coupling, EmptyCoupling
from transformers import AutoTokenizer
from datasets import GoedelDataset
import torch

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
        contexts = []
        for i in range(len(batch)):
            context = batch[i]['context']
            prev_attempt = batch[i].get('prev_attempt', '')
            target = batch[i]['target']

            context_ids = self.tokenizer(context, return_tensors='pt').input_ids[:self.max_len]
            context_len = context_ids.shape[1]

            prev_ids = self.tokenizer(prev_attempt,  return_tensors='pt').input_ids[:self.max_len - context_len]
            target_ids = self.tokenizer(target,  return_tensors='pt').input_ids[:self.max_len - context_len]

            # combine context and prev_attempt for x0, context and target for x1:
            # x0 = torch.cat((context_ids.squeeze(0), prev_ids.squeeze(0)[1:]), dim=0)[:self.max_len]
            # x1 = torch.cat((context_ids.squeeze(0), target_ids.squeeze(0)[1:]), dim=0)[:self.max_len]

            x0s.append(prev_ids)
            x1s.append(target_ids)
            context_lens.append(context_len)
            contexts.append(context_ids.squeeze(0))

        # assumes that the alignment will keep context at the start for all examples so we can use context_lens
        # todo collate batch only with x1s, x0s after context, add context to batch instead
        ret =  collate_batch_goedel(x1s, x0s, pad_token=self.tokenizer.pad_token_id, gap_token=151651)
        ret['context_lens'] = context_lens
        ret['contexts'] = contexts

        return ret

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self._collate, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size,
                          collate_fn=self._collate, num_workers=0)
