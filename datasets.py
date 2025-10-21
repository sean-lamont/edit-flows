import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Callable
from utils import *
import glob
import json

class GoedelDataset(Dataset):
    def __init__(self, folder_path='processed_data'):
        self.files = glob.glob(f'{folder_path}/*.jsonl')
        self.data = []

        for file_path in self.files:
            with open(file_path, 'r') as f:
                for line in f:
                    sample = json.loads(line)
                    if sample['type'] == 'correction':
                        self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]
