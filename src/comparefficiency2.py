
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import sys
import random
import time
from datetime import datetime

from src.conv_conventional import Imbouttathrowhands
from datasets.cache_loader import SymbolicDataset


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42

set_seed(seed)


def run_mini_train(model, train_loader, val_loader, epochs=30, lr=0.001):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-2
    )

    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    best_val = 0
    patience = 30
    trigger_time = 0
    epoch_times = []
    for epoch in range(epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()

        # TRAIN
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total

        # VALIDATION
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        if val_acc > best_val:
            best_val = val_acc
            trigger_time = 0
        else:
            trigger_time += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_times.append(epoch_duration)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% | "
            f"Epoch Duration: {epoch_duration:.2f}s"
        )

        if trigger_time >= patience:
            print(f"Stopping early at epoch {epoch+1} - no improvement.")
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            return best_val, avg_epoch_time

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    return best_val, avg_epoch_time


def save_results_csv(dataset_name,seed, percentages,symbolic_results, raw_results, s_epoch_times, r_epoch_times):
    assets_dir = os.path.join("assets")
    os.makedirs(assets_dir, exist_ok=True)
    seed = str(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"rgb_comparison_{dataset_name}_{seed}_{timestamp}.csv"
    filepath = os.path.join(assets_dir, filename)

    df = pd.DataFrame({
        "training_data_percent": [p * 100 for p in percentages],
        "symbolic_val_accuracy": symbolic_results,
        "raw_rgb_val_accuracy": raw_results,
        "average_epoch_time_symbolic" : s_epoch_times,
        "average_epcoh_time_raw" : r_epoch_times
    })

    df.to_csv(filepath, index=False)

    print(f"\nSaved results to: {filepath}")


def main():
    dataset_name = sys.argv[1] if len(sys.argv) == 2 else "imagenette2"

    raw_root = os.path.join("datasets", dataset_name)
    cache_root = os.path.join("datasets", f"symbolic_cache_{dataset_name}")

    val_folder = "test" if "caltech" in dataset_name else "val"

    percentages = [ 0.005, 0.0075, 0.01, 0.02, 0.05]

    symbolic_results = []
    raw_results = []
    average_epoch_times_s = []
    average_epoch_times_r = []
    # SYMBOLIC DATASETS
    s_train_full = SymbolicDataset(
        os.path.join(cache_root, "train/processed_data.pt"),
        os.path.join(cache_root, "train/labels.pt")
    )

    s_val_full = SymbolicDataset(
        os.path.join(cache_root, f"{val_folder}/processed_data.pt"),
        os.path.join(cache_root, f"{val_folder}/labels.pt")
    )

    # RAW DATASETS
    raw_transform = transforms.Compose([
        transforms.Resize((149, 149)),
        transforms.CenterCrop((149, 149)),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
    ])

    r_train_full = datasets.ImageFolder(
        root=os.path.join(raw_root, "train"),
        transform=raw_transform
    )

    r_val_full = datasets.ImageFolder(
        root=os.path.join(raw_root, val_folder),
        transform=raw_transform
    )

    if "caltech" in dataset_name.lower():
        target_classes = r_train_full.classes[:30]

        train_indices = [
            i for i, label in enumerate(r_train_full.targets)
            if r_train_full.classes[label] in target_classes
        ]

        val_indices = [
            i for i, label in enumerate(r_val_full.targets)
            if r_val_full.classes[label] in target_classes
        ]

        r_train_full = Subset(r_train_full, train_indices)
        r_val_full = Subset(r_val_full, val_indices)

    

    num_classes = len(torch.unique(s_train_full.labels))

    print(f"--- Dataset: {dataset_name} ---")
    print(f"Synced classes: {num_classes}")
    print(f"Total training pool: {len(s_train_full)} images")

    for p in percentages:
        print(f"\n--- Testing with {p*100}% of data ---")

        train_pool_size = len(s_train_full)
        train_sz = int(train_pool_size * p)
        val_sz = min(500, len(s_val_full))

        indices_train = np.random.choice(
            train_pool_size,
            train_sz,
            replace=False
        )

        indices_val = np.random.choice(
            len(s_val_full),
            val_sz,
            replace=False
        )

        # SYMBOLIC
        print("Running Symbolic model (1 channel)...")


        s_loader = DataLoader(
            Subset(s_train_full, indices_train),
            batch_size=16,
            shuffle=True
        )

        sv_loader = DataLoader(
            Subset(s_val_full, indices_val),
            batch_size=16,
            shuffle=False
        )

        s_model = Imbouttathrowhands(
            num_classes=num_classes,
            in_channels=1
        )

        s_acc, s_epoch_time = run_mini_train(
            s_model,
            s_loader,
            sv_loader
        )

        symbolic_results.append(s_acc)
        average_epoch_times_s.append(s_epoch_time)

        # RAW RGB
        print("Running Raw model (3 channels)...")

        r_loader = DataLoader(
            Subset(r_train_full, indices_train),
            batch_size=16,
            shuffle=True
        )

        rv_loader = DataLoader(
            Subset(r_val_full, indices_val),
            batch_size=16,
            shuffle=False
        )

        r_model = Imbouttathrowhands(
            num_classes=num_classes,
            in_channels=3
        )

        r_acc, r_epoch_time = run_mini_train(
            r_model,
            r_loader,
            rv_loader
        )

        raw_results.append(r_acc)
        average_epoch_times_r.append(r_epoch_time)

 
    save_results_csv(
        dataset_name,
        seed,
        percentages,
        symbolic_results,
        raw_results, 
        average_epoch_times_s,
        average_epoch_times_r
    )


if __name__ == "__main__":
    main()