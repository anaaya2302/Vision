import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


transform = transforms.Compose([
    transforms.Resize((149, 149)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


baseline_model = models.resnet18(weights=None)
baseline_model.fc = torch.nn.Linear(baseline_model.fc.in_features, 10) 

device = torch.device("cuda")
baseline_model.to(device)


def main():
    # 1. Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 128  # 4090 can handle this easily
    epochs = 5
    lr = 1e-3
    num_classes = 10

    # 2. Transforms (Match your symbolic resolution!)
    transform = transforms.Compose([
        transforms.Resize((149, 149)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # 3. Data Loaders (Points to your raw ImageNet images)
    train_ds = datasets.ImageFolder(root="datasets/imagenette2/train", transform=transform)
    val_ds = datasets.ImageFolder(root="datasets/imagenette2/val", transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 4. Model (Standard ResNet-18, no pre-training)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # 5. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 6. Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for X, y in loop:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == y).sum().item()
            train_total += y.size(0)
            loop.set_postfix(loss=loss.item(), acc=train_correct/train_total)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                val_correct += (outputs.argmax(1) == y).sum().item()
                val_total += y.size(0)

        print(f"\nSummary Epoch {epoch+1}:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_correct/train_total:.4f}")
        print(f"Val Loss:   {val_loss/len(val_loader):.4f} | Val Acc:   {val_correct/val_total:.4f}\n")

if __name__ == "__main__":
    main()