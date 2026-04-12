import torch.nn as nn
import torch
import tqdm
class PotatoClassifier(nn.Module):
    def __init__(self, num_classes=10, num_filters=8):
        super(PotatoClassifier, self).__init__()
        # Layer 1 ONLY
        self.conv1 = nn.Conv2d(2, num_filters, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # Global Average Pool: flattens any spatial size to (num_filters x 1 x 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final Linear
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        x = x.float()
        x = self.conv1(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1) # Flatten
        return self.fc(x)
    
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