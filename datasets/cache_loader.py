#Had AI code it. I have zero clue what's happening tbh.
import torch
from torch.utils.data import Dataset
import os

class SymbolicDataset(Dataset):
    def __init__(self, tensor_path, labels_path=None):
        print(f"Loading consolidated tensor from {tensor_path}...")
        # Use mmap=True to keep RAM usage low on your laptop
        self.data = torch.load(tensor_path, weights_only=True, mmap=True)
        
        if labels_path and os.path.exists(labels_path):
            self.labels = torch.load(labels_path, weights_only=True)
        else:
            print("WARNING: No labels found. Using dummy zeros.")
            self.labels = torch.zeros(self.data.shape[0], dtype=torch.long)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # 1. Grab the slice
        index_map = self.data[idx].long()
        
        # 2. THE FIX: Squeeze out ALL extra dimensions of size 1
        # This turns [1, 1, 149, 149] or [1, 149, 149] into just [149, 149]
        index_map = index_map.squeeze()
        
        label = self.labels[idx]
        
        return index_map, label