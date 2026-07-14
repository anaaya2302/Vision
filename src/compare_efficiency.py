import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import sys

from src.conv_conventional import ShapeClassifier 
from datasets.cache_loader import SymbolicDataset

def run_mini_train(model, train_loader, val_loader, epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val = 0
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
          
            inputs, labels = inputs.to(device).float(), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device).float(), labels.to(device)
                outputs = model(inputs)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total if total > 0 else 0
        best_val = max(best_val, acc)
    return best_val

def main():

    dataset_name = sys.argv[1] if len(sys.argv) == 2 else "imagenette2"
    
    raw_root = os.path.join("datasets", dataset_name)
    cache_root = os.path.join("datasets", f"symbolic_cache_{dataset_name}")
    
    # Define splits based on dataset convention
    val_folder = "test" if "caltech" in dataset_name else "val"
    
    percentages = [0.01, 0.05] 
    symbolic_results, raw_results = [], []

    # 1. SETUP DATASETS
    # Symbolic
    s_train_full = SymbolicDataset(os.path.join(cache_root, "train/processed_data.pt"), 
                                  os.path.join(cache_root, "train/labels.pt"))
    s_val_full = SymbolicDataset(os.path.join(cache_root, f"{val_folder}/processed_data.pt"), 
                                os.path.join(cache_root, f"{val_folder}/labels.pt"))
    
    # Raw
    raw_transform = transforms.Compose([
        transforms.Resize((149, 149)), 
        transforms.CenterCrop((149, 149)),
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.ToTensor(),
    ])
    
    r_train_full = datasets.ImageFolder(root=os.path.join(raw_root, "train"), transform=raw_transform)
    r_val_full = datasets.ImageFolder(root=os.path.join(raw_root, val_folder), transform=raw_transform)

    # Automatically detect class count
    num_classes = len(r_train_full.classes)
    print(f"Detected {num_classes} classes for {dataset_name}")

    for p in percentages:
        print(f"\n--- Testing with {p*100}% of data ---")
        
        train_sz = int(len(r_train_full) * p)
        val_sz = min(200, len(r_val_full))
        
        indices_train = np.random.choice(len(r_train_full), train_sz, replace=False)
        indices_val = np.random.choice(len(r_val_full), val_sz, replace=False)

        # --- RUN SYMBOLIC MODEL ---
        print(f"Running Symbolic model (2 channels)...")
        s_loader = DataLoader(Subset(s_train_full, indices_train), batch_size=16, shuffle=True)
        sv_loader = DataLoader(Subset(s_val_full, indices_val), batch_size=16, shuffle=False)
        
        # Passing in_channels=2 for Trig Tokens
        s_model = ShapeClassifier(num_classes=num_classes, in_channels=2)
        s_acc = run_mini_train(s_model, s_loader, sv_loader)
        symbolic_results.append(s_acc)

        # --- RUN RAW MODEL ---
        print(f"Running Raw model (3 channels)...")
        r_loader = DataLoader(Subset(r_train_full, indices_train), batch_size=16, shuffle=True)
        rv_loader = DataLoader(Subset(r_val_full, indices_val), batch_size=16, shuffle=False)
        
        # Passing in_channels=3 for RGB
        r_model = ShapeClassifier(num_classes=num_classes, in_channels=3)
        r_acc = run_mini_train(r_model, r_loader, rv_loader)
        raw_results.append(r_acc)
        
        print(f"Results for {p*100}%: Symbolic {s_acc:.4f} | Raw {r_acc:.4f}")

    # 2. PLOT
    plt.figure(figsize=(10, 6))
    plt.plot([p*100 for p in percentages], symbolic_results, 'o-', label='Symbolic (Trig Tokens)')
    plt.plot([p*100 for p in percentages], raw_results, 's--', label='Raw Pixels')
    plt.title(f'Data Efficiency on {dataset_name}')
    plt.xlabel('Training Data (%)')
    plt.ylabel('Val Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'efficiency_{dataset_name}.png')
    plt.show()

if __name__ == "__main__":
    main()