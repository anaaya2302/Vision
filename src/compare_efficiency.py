import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Import your architectures
from src.conv import ShapeClassifier as SymbolicModel

from datasets.cache_loader import SymbolicDataset

def run_mini_train(model, train_loader, val_loader, epochs=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_val = 0
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        
        # Validation on the small subset
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        
        acc = correct / total
        best_val = max(best_val, acc)
    return best_val

def main():
    percentages = [0.01, 0.05, 0.10, 0.25] 
    symbolic_results = []
    raw_results = []

    # 1. SETUP DATASETS
    # Symbolic (Trig Tokens)
    s_train_full = SymbolicDataset("datasets/symbolic_cache/train/processed_data.pt", "datasets/symbolic_cache/train/labels.pt")
    s_val_full = SymbolicDataset("datasets/symbolic_cache/val/processed_data.pt", "datasets/symbolic_cache/val/labels.pt")
    
    # Raw Grayscale (Pixels)
    raw_transform = transforms.Compose([
        transforms.Resize((149, 149)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    r_train_full = datasets.ImageFolder(root='datasets/imagenette2/train', transform=raw_transform)
    r_val_full = datasets.ImageFolder(root='datasets/imagenette2/val', transform=raw_transform)

    for p in percentages:
        print(f"\n--- Testing with {p*100}% of data ---")
        
        # Subset Sizes
        train_sz = int(len(r_train_full) * p)
        val_sz = 200 # Fixed small val set to prevent hanging
        
        indices_train = np.random.choice(len(r_train_full), train_sz, replace=False)
        indices_val = np.random.choice(len(r_val_full), val_sz, replace=False)

        # --- RUN SYMBOLIC MODEL ---
        print(f"Running Symbolic model...")
        s_loader = DataLoader(Subset(s_train_full, indices_train), batch_size=16, shuffle=True, num_workers=0)
        sv_loader = DataLoader(Subset(s_val_full, indices_val), batch_size=16, shuffle=False, num_workers=0)
        
        s_model = SymbolicModel(num_classes=10)
        # Ensure Symbolic model is expecting 2 channels (sin/cos)
        s_acc = run_mini_train(s_model, s_loader, sv_loader)
        symbolic_results.append(s_acc)
        print(f"Symbolic Best Acc: {s_acc:.4f}")

        # --- RUN GRAYSCALE MODEL ---
        print(f"Running Grayscale model...")
        r_loader = DataLoader(Subset(r_train_full, indices_train), batch_size=16, shuffle=True, num_workers=0)
        rv_loader = DataLoader(Subset(r_val_full, indices_val), batch_size=16, shuffle=False, num_workers=0)
        
        r_model = SymbolicModel(num_classes=10)
        r_model.layer1[0] = nn.Conv2d(1, 32, stride=2, kernel_size=2, padding=1)
        
        r_acc = run_mini_train(r_model, r_loader, rv_loader)
        raw_results.append(r_acc)
        print(f"Grayscale Best Acc: {r_acc:.4f}")

    # 2. PLOT THE DATA EFFICIENCY CURVE
    plt.figure(figsize=(10, 6))
    plt.plot([p*100 for p in percentages], symbolic_results, 'o-', label='Symbolic (Trig Tokens)', color='blue', linewidth=2)
    plt.plot([p*100 for p in percentages], raw_results, 's--', label='Raw Grayscale Pixels', color='red', linewidth=2)
    
    plt.xlabel('Percentage of Training Data (%)')
    plt.ylabel('Validation Accuracy (Subset)')
    plt.title('Data Efficiency: Symbolic vs. Raw Pixels')
    plt.legend()
    plt.grid(True)
    plt.savefig('efficiency_comparison.png')
    print("\nPlot saved as efficiency_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()