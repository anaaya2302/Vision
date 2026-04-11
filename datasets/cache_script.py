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
        sys.exit(f"Error: {split_dir} not found. Check your folder structure!")

    batch_size = 64
    transform = transforms.Compose([
        transforms.Resize((447, 447)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=split_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    edge_templates = Preprocessor.get_edge_templates(device)

    print(f"\n[{split}] Starting processing ({len(dataset)} images)...")

    # Save labels
    all_labels = []
    for _, labels in tqdm(loader, desc=f"[{split}] Labels"):
        all_labels.append(labels.cpu())
    torch.save(torch.cat(all_labels, dim=0), os.path.join(cache_dir, "labels.pt"))

    # Save image order
    image_paths = [s[0] for s in dataset.samples]
    with open(os.path.join(cache_dir, "image_order.txt"), "w") as f:
        for p in image_paths:
            f.write(f"{os.path.relpath(p, split_dir)}\n")

    # Process and save features
    all_indices = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=f"[{split}] Features"):
            images = images.to(device)
            images = Preprocessor.luminance(images, channels_last=False, normalised=False)
            edge_grads = Preprocessor.edge_grads(images)
            _, edge_probs = Preprocessor.edge_probs(*edge_grads)
            edge_binary = Preprocessor.thresholding_grayscale(edge_probs).float()
            print(edge_binary.device, edge_templates.device)
            best_match = Preprocessor.template_match(edge_binary, edge_templates)
            all_indices.append(best_match.detach().cpu().byte())

    print(f"[{split}] Consolidating...")
    final_tensor = torch.cat(all_indices, dim=0)
    tensor_save_path = os.path.join(cache_dir, "processed_data.pt")
    torch.save(final_tensor, tensor_save_path)

    final_size_gb = os.path.getsize(tensor_save_path) / (1024**3)
    print(f"[{split}] Done! {tensor_save_path} ({final_size_gb:.2f} GB)")


def main():
    print("GPU is active:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        sys.exit("GPU is throwing a tantrum. Use a machine with CUDA.")

    device = torch.device("cuda")
    for split in ["train", "val"]:
        cache_split(split, device)


if __name__ == "__main__":
    main()