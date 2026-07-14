import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import your model and training function
from src.conv import ShapeClassifier, train_model

def main():
    # 1. Standard RGB Transformations
    transform = transforms.Compose([
        transforms.Resize((149, 149)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
         # Standard ImageNet scaling
    ])

    # 2. Load Raw Imagenette
    train_ds = datasets.ImageFolder(root='datasets/imagenette2/train', transform=transform)
    val_ds = datasets.ImageFolder(root='datasets/imagenette2/val', transform=transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    # 3. Initialize Model and MODIFY THE FIRST LAYER
    model = ShapeClassifier(num_classes=10)
    
    # OVERRIDE: Change input channels from 2 to 3
    # We keep the stride=2, kernel_size=2, and padding=1 you just optimized
    model.layer1[0] = nn.Conv2d(1, 32, stride=2, kernel_size=2, padding=1)

    print("--- Starting RAW RGB Baseline Run ---")
    print(f"Input Shape: (3, 149, 149) | Data: Raw RGB")
    
    # 4. Train
    train_model(model, train_loader, val_loader, epochs=30, lr=1e-4)

    torch.save(model.state_dict(), 'raw_rgb_baseline.pth')

if __name__ == "__main__":
    main()