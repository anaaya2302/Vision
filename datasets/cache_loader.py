#Had AI code it. I have zero clue what's happening tbh.
import torch
from torch.utils.data import Dataset, DataLoader
import os

class SymbolicDataset(Dataset):
    def __init__(self, tensor_path, labels_path=None):
        # mmap=True is crucial for laptops: it keeps the file on disk and 
        # only loads the specific 'idx' into RAM when requested.
        self.data = torch.load(tensor_path, weights_only=True, mmap=True)
        
        if labels_path and os.path.exists(labels_path):
            self.labels = torch.load(labels_path, weights_only=True)
        else:
            self.labels = torch.zeros(self.data.shape[0], dtype=torch.long)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # 1. Grab the trig features. 
        # If cached as (N, 2, 149, 149), self.data[idx] is (2, 149, 149)
        x = self.data[idx]
        
        # 2. Grab the label
        y = self.labels[idx]
        
        return x, y


    
