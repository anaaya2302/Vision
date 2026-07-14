import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from src.data import Preprocessor
import sys


def cache_split(raw_data_root, cache_root, split, device, target_dataset, img_size=(447, 447), num_class=30):

    split_dir = os.path.join(raw_data_root, split)
    cache_dir = os.path.join(cache_root, split)
    os.makedirs(cache_dir, exist_ok=True)

    if not os.path.exists(split_dir):
        print(f"Skipping: {split_dir} not found.")
        return
    
    

    batch_size = 64
    # Ensure 3 channels
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size), # Keeps aspect ratio sane
        transforms.Lambda(lambda x: x.convert('RGB')), 
        transforms.ToTensor()
    ])

    full_dataset = datasets.ImageFolder(root=split_dir, transform=transform)

    if "caltech" in target_dataset.lower():
        
        target_classes = full_dataset.classes[:num_class]
        indices = [i for i, label in enumerate(full_dataset.targets) 
                   if full_dataset.classes[label] in target_classes]
        dataset = torch.utils.data.Subset(full_dataset, indices)
        print(f"[{split}] Subset created: Using first {num_class} classes of Caltech.")
    else:
        # All of Imagenette
        dataset = full_dataset
        print(f"[{split}] Using full dataset ({len(full_dataset.classes)} classes).")
    
   

   
    
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=False)

    edge_templates = Preprocessor.get_edge_templates(device)
    

    print(f"\n[{split}] Processing {raw_data_root}...")

    all_labels = []
    all_features = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"[{split}] Processing"):
           
            all_labels.append(labels.cpu())

           
            images = images.to(device)
            images = Preprocessor.luminance(images, channels_last=False, normalise=False)
            edge_grads = Preprocessor.edge_grads(images)
            _, edge_probs = Preprocessor.edge_probs(*edge_grads)
            edge_binary = Preprocessor.thresholding_grayscale(edge_probs).float()
            
            best_match = Preprocessor.template_match(edge_binary, edge_templates) 
            
            

            all_features.append(best_match.cpu().half())
    torch.save(torch.cat(all_labels, dim=0), os.path.join(cache_dir, "labels.pt"))
    torch.save(torch.cat(all_features, dim=0), os.path.join(cache_dir, "processed_data.pt"))
    print(f"[{split}] Saved to {cache_dir}")

def main():
    # Isn't defensive. for now just check the spelling of the command line argument. pls.
    assert len(sys.argv) == 2, "specify dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    target_dataset = sys.argv[1] 
    
    raw_path = os.path.join("datasets", target_dataset)
    cache_path = os.path.join("datasets", f"symbolic_cache_{target_dataset}")

    # Determine splits based on dataset name
    splits = ["train", "test"] if "caltech" in target_dataset else ["train", "val"]

    for split in splits:
        cache_split(raw_path, cache_path, split, device, target_dataset)

if __name__ == "__main__":
    main()