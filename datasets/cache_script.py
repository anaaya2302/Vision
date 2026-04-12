import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from src.data import Preprocessor
import sys

def cache_split(split, device):
    raw_data_root = os.path.join("datasets", "imagenette2")
    split_dir = os.path.join(raw_data_root, split)
    cache_dir = os.path.join("datasets", "symbolic_cache", split)
    os.makedirs(cache_dir, exist_ok=True)

    if not os.path.exists(split_dir):
        sys.exit(f"Error: {split_dir} not found.")

    batch_size = 64
    transform = transforms.Compose([
        transforms.Resize((447, 447)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=split_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    edge_templates = Preprocessor.get_edge_templates(device)
    angle_map = Preprocessor.get_angle_mapping(device)

    print(f"\n[{split}] into the trenches we go!")

    
    all_labels = []
    for _, labels in tqdm(loader, desc=f"[{split}] Labels"):
        all_labels.append(labels)
    torch.save(torch.cat(all_labels, dim=0), os.path.join(cache_dir, "labels.pt"))

    # Process and Save Trig Features
    all_features = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=f"[{split}] Trig Features"):
            images = images.to(device)
            images = Preprocessor.luminance(images, channels_last=False, normalise=False)
            edge_grads = Preprocessor.edge_grads(images)
            _, edge_probs = Preprocessor.edge_probs(*edge_grads)
            edge_binary = Preprocessor.thresholding_grayscale(edge_probs).float()
            
            # Get discrete index (0-8)
            best_match = Preprocessor.template_match(edge_binary, edge_templates) 
            

            
            # Pull (sin, cos) and reshape back to (N, 2, H/3, W/3)
            trig_features = Preprocessor.idx_to_trig(best_match, angle_map)
            
            all_features.append(trig_features.cpu().half())

    print(f"[{split}] Consolidating tensors...")
    final_tensor = torch.cat(all_features, dim=0)
    save_path = os.path.join(cache_dir, "processed_data.pt")
    torch.save(final_tensor, save_path)
    print(f"[{split}] Done! Saved to {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for split in ["train", "val"]:
        cache_split(split, device)

if __name__ == "__main__":
    main()
