import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Callable
import glob
import json

class GoedelDataset(Dataset):
    def __init__(self, folder_path='processed_data_code_only'):
        self.files = glob.glob(f'{folder_path}/*.jsonl')
        self.data = []

        for file_path in self.files:
            with open(file_path, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    # if sample['type'] == 'correction':
                    if len(sample['target']) > 10:
                        if len(sample['prev_attempt']) < 2:
                            sample['prev_attempt'] = 'Initial Attempt'
                        self.data.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        ret = self.data[idx]
        ret['idx'] = idx
        return ret
