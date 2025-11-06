import torch
from torch.utils.data import Dataset
import goedel_dataset  # <-- NEW
import os
import datasets


class GoedelDataset(Dataset):
    def __init__(self, name='sean-lamont/goedel_preprocessed'):
        # if not os.path.exists(folder_path):
        #     raise FileNotFoundError(
        #         f"Hugging Face dataset not found at {folder_path}. "
        #         "Did you run the preprocess.py script?"
        #     )
        #
        # print(f"Loading precomputed dataset from {folder_path}...")
        # self.dataset = datasets.load_from_disk(folder_path)

        self.dataset = datasets.load_dataset(name)

        # only load correction examples for now
        self.dataset = self.dataset.filter(lambda example: example['type'] == 'correction')

        print("Dataset loaded.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Returns a single sample from the Hugging Face dataset.
        This will be a dictionary where values are Python lists,
        not tensors.
        """
        return self.dataset[idx]