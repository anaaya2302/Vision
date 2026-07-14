import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets.cache_loader import SymbolicDataset

# IT ALSO MEMORIZED ALL THE DATA

class LinearClassifier(nn.Module):
    def __init__(self, input_channels=2, grid_size=149, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.dropout = nn.Dropout(p=0.5)
        # Total input features: 2 * 149 * 149 = 44,402
        self.input_dim = input_channels * grid_size * grid_size
        
        # A single linear layer (Logistic Regression)
        self.classifier = nn.Linear(self.input_dim, num_classes)

    def forward(self, x):
        # x shape: (Batch, 2, 149, 149)
        # Flatten to: (Batch, 44402)
        x = x.view(x.size(0), -1)

        x = x.float()
        return self.classifier(x)
    
def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, weight_decay=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Weight decay (L2 regularization) prevents the model from relying on noise
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss, train_correct = 0.0, 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
        
        # --- Validation Phase ---
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        
        train_acc = train_correct / len(train_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        
        print(f"Loss: {train_loss/len(train_loader.dataset):.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        

def main():
    # 1. Setup paths (Change these to your cache locations)
    train_path = "datasets/symbolic_cache/train/processed_data.pt"
    train_labels = "datasets/symbolic_cache/train/labels.pt"
    val_path = "datasets/symbolic_cache/val/processed_data.pt"
    val_labels = "datasets/symbolic_cache/val/labels.pt"
    
    # 2. Initialize Datasets and Loaders
    # We use mmap=True to keep your laptop RAM usage low
    train_ds = SymbolicDataset(train_path, train_labels)
    val_ds = SymbolicDataset(val_path, val_labels)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, shuffle=False)
    
    # 3. Initialize Model
    model = LinearClassifier(num_classes=10)
    
    # 4. Run Training
    # If it still overfits, increase weight_decay to 1e-3 or higher.
    train_model(model, train_loader, val_loader, epochs=30, lr=1e-4, weight_decay=1e-3)

if __name__ == "__main__":
    main()