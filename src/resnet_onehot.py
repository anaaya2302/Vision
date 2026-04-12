import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from datasets.cache_loader import SymbolicDataset
from tqdm import tqdm


#RESNETS WERE WAY TOO BIG. no matter what i did it overfit like crazy.


class OneHotResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        #Add dropout, scheduler, 
        self.embedding = nn.Embedding(9, 64)
        
        # All hail torch code that I barely understand
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.bn1 = nn.BatchNorm2d(64)
        # Can't use standard stem... the downsampling would nuke my preprocessor

        self.resnet.maxpool = nn.Identity()
        
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, X):
        # X is my cached preprocessed input 
        # By my arithemetic, it should be [N, 149, 149]



        X = self.embedding(X.long()) # Turn indices into 64-dim vectors
        X = X.permute(0, 3, 1, 2)
        
        # These floats and bytes are literally waiting to throw an error I swear

        X = X.float()
        
        # Yeet into ResNet
        X = self.resnet(X)
        return X
    


def main():
    device = torch.device("cuda")

    train_ds = SymbolicDataset(
        tensor_path="datasets/symbolic_cache/train/processed_data.pt", 
        labels_path="datasets/symbolic_cache/train/train_labels.pt"
    )

    val_ds = SymbolicDataset(
        tensor_path="datasets/symbolic_cache/val/processed_data.ptS", 
        labels_path="datasets/symbolic_cache/val/val_labels.pt"
    )


    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    model = OneHotResNet(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(20):
        # Training
        model.train()
        train_loss, correct, total = 0, 0, 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1} train"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        print(f"Train loss: {train_loss/len(train_loader):.4f} | Acc: {correct/total:.4f}")

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for X, y in tqdm(val_loader, desc=f"Epoch {epoch+1} val"):
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                val_loss += loss.item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        print(f"Val loss:   {val_loss/len(val_loader):.4f} | Acc: {correct/total:.4f}")

if __name__ == "__main__":
    main()