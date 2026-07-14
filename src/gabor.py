import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import time
import math
from datetime import datetime
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from src.comparefficiency2 import run_mini_train

# Assuming these exist based on your provided snippets
from src.conv_conventional import Imbouttathrowhands
import torch
import torch.nn as nn
import torch.nn.functional as F

class StructuredGaborBank(nn.Module):
    def __init__(self, in_channels=1, kernel_size=7):
        super().__init__()
        # We want to mimic your 9 templates: 
        # (Horizontal, Vertical, Diagonal) x (3 spatial offsets/phases)
        thetas = [0, torch.pi/2, torch.pi/4] 
        phases = [-torch.pi/2, 0, torch.pi/2] # Mimics Left, Center, Right
        
        kernels = []
        sigma = 1.5
        lambd = 4.0
        gamma = 0.5
        
        y, x = torch.meshgrid(
            torch.linspace(-3, 3, kernel_size),
            torch.linspace(-3, 3, kernel_size),
            indexing='ij'
        )

        for th in thetas:
            for ps in phases:
                x_theta = x * torch.cos(torch.tensor(th)) + y * torch.sin(torch.tensor(th))
                y_theta = -x * torch.sin(torch.tensor(th)) + y * torch.cos(torch.tensor(th))
                
                # Gabor Function
                gb = torch.exp(-0.5 * (x_theta**2 / sigma**2 + (gamma**2 * y_theta**2) / sigma**2)) * \
                     torch.cos(2 * torch.pi * x_theta / lambd + ps)
                kernels.append(gb.unsqueeze(0).unsqueeze(0))

        self.register_buffer('weight', torch.cat(kernels, dim=0))

    def forward(self, x):
        # Convert RGB to Luminance inside the model for parity with your Preprocessor
        if x.shape[1] == 3:
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Apply the 9 Gabor filters
        return F.conv2d(x, self.weight, padding=self.weight.shape[2]//2)

class GaborModel(nn.Module):
    def __init__(self, num_classes=30):
        super().__init__()
        self.frontend = StructuredGaborBank()
        # We pass 9 channels to your backbone. 
        # Since these are floats, in_channels=9 will trigger the Conv2d stem 
        # rather than the Embedding stem in your 'Imbouttathrowhands' class.
        self.backbone = Imbouttathrowhands(num_classes=num_classes, in_channels=9)

    def forward(self, x):
        features = self.frontend(x)
        return self.backbone(features)
    
class GaborBankModel(nn.Module):
    def __init__(self, num_classes):
        super(GaborBankModel, self).__init__()
        self.gabor_frontend = StructuredGaborBank(in_channels=1)
        # The frontend outputs 9 channels, so our classifier must accept 9 in_channels
        self.backbone = Imbouttathrowhands(num_classes=num_classes, in_channels=9)

    def forward(self, x):
        x = self.gabor_frontend(x)
        x = self.backbone(x)
        return x

def main():
    dataset_name = "imagenette2"
    raw_root = os.path.join("datasets", dataset_name)
    
    # Standard transforms for Imagenette2
    raw_transform = transforms.Compose([
        transforms.Resize((149, 149)),
        transforms.CenterCrop((149, 149)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(raw_root, "train"), transform=raw_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(raw_root, "val"), transform=raw_transform)
    
    num_classes = len(train_dataset.classes)
    percentages = [0.001, 0.005, 0.0075, 0.01, 0.02, 0.05]
    
    results, times = [], []

    for p in percentages:
        print(f"\n--- Gabor Experiment: {p*100}% Data ---")
        indices = np.random.choice(len(train_dataset), int(len(train_dataset) * p), replace=False)
        train_loader = DataLoader(Subset(train_dataset, indices), batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        model = GaborBankModel(num_classes=num_classes)
        # Use your existing run_mini_train function
        acc, avg_time = run_mini_train(model, train_loader, val_loader)
        
        results.append(acc)
        times.append(avg_time)

    # Save to CSV
    df = pd.DataFrame({
        "percent": [p*100 for p in percentages],
        "gabor_val_acc": results,
        "avg_epoch_time": times
    })
    os.makedirs("assets", exist_ok=True)
    df.to_csv(f"assets/gabor_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

if __name__ == "__main__":
    main()