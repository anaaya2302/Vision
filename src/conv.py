import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets.cache_loader import SymbolicDataset

class ShapeClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        

        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 32, stride=2, kernel_size=2, padding=1), #downsampled to 74
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
        )
        
        
        # Layer 2: Combine edges into complex shapes
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2) # Down to 37x37
        )
        
        # Layer 3: High-level features
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.float()
        x = self.layer1(x)
        x = self.layer2(x)
       
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
    
# --- 3. Training Function ---
def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Weight decay here helps prevent even these few params from over-focusing
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    print(f"Training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            
        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        
        t_acc = train_correct / len(train_loader.dataset)
        v_acc = val_correct / len(val_loader.dataset)
        print(f"Loss: {train_loss/len(train_loader.dataset):.4f} | Train Acc: {t_acc:.4f} | Val Acc: {v_acc:.4f}")



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
    model = ShapeClassifier(num_classes=10)
    
    # 4. Run Training
    # If it still overfits, increase weight_decay to 1e-3 or higher.
    train_model(model, train_loader, val_loader, epochs=30, lr=1e-4)

    torch.save(model.state_dict(), 'patch_logic_48_38.pth'); print("SAVED. You can breathe now.")

# --- 4. Main Script ---
if __name__ == "__main__":
    main()