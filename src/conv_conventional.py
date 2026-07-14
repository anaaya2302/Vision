import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datasets.cache_loader import SymbolicDataset
import os
import torch.nn.functional as F


class Imbouttathrowhands(nn.Module):
    def __init__(self, num_classes=30, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        
        # 1. STEM
        if in_channels == 1: 
            self.stem = nn.Embedding(9, 16, padding_idx=0) 
            self.proj = nn.Conv2d(16, 16, kernel_size=1)
        else:
            # keeping params similar
            self.stem = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)

      
        self.conv_vertical = nn.Conv2d(16, 16, kernel_size=(7, 1), padding=(6, 0),dilation=2)
        self.conv_horizontal = nn.Conv2d(16, 16, kernel_size=(1, 7), padding=(0, 6), dilation=2)
        self.bn1 = nn.BatchNorm2d(32) 

        self.conv_vertical_2 = nn.Conv2d(32, 32, kernel_size=(7, 1), padding=(6, 0), dilation=2)
        self.conv_horizontal_2 = nn.Conv2d(32, 32, kernel_size=(1, 7), padding=(0, 6), dilation=2)
        self.bn2 = nn.BatchNorm2d(64)
        

        self.relu = nn.LeakyReLU(0.1)
       
        self.merger = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )


        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1)
        )

        # 5. CLASSIFIER
        self.zone_pool = nn.AdaptiveAvgPool2d((6, 6)) 
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 128), 
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Stem Logic
        if self.in_channels == 1:
            x = x.squeeze(1).long() 
            x = self.stem(x).permute(0, 3, 1, 2)
            x = self.proj(x)
        else:
            x = self.stem(x)
            
        # Asymmetric Logic
        h = self.conv_horizontal(x)
        v = self.conv_vertical(x)
        x = torch.cat([h, v], dim=1)
        x = self.relu(self.bn1(x))
        h2 = self.conv_horizontal_2(x)
        v2 = self.conv_vertical_2(x)
        x = torch.cat([h2, v2], dim=1)
        x = self.relu(self.bn2(x))


        
        x = self.merger(x)
        x = self.layer2(x) 
        
        x = self.zone_pool(x)
        return self.classifier(x)

    
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

        best_v_acc = 0.0
    
    
        if v_acc > best_v_acc:
            best_v_acc = v_acc
        
            save_dir = "checkpoints"
            os.makedirs(save_dir, exist_ok=True) # Creates the folder if it's not there

            model_path = os.path.join(save_dir, "best_symbolic_model.pth")
            torch.save(model.state_dict(), model_path)


    return best_v_acc



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
    model = Imbouttathrowhands(num_classes=30)
    
    # 4. Run Training
    # If it still overfits, increase weight_decay to 1e-3 or higher.
    best_val_acc = train_model(model, train_loader, val_loader, epochs=30, lr=1e-4)
    print(best_val_acc)
    

# --- 4. Main Script ---
if __name__ == "__main__":
    main()