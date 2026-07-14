import os
import shutil
from tqdm import tqdm

def physical_split():
    # Im sorry for the hardcoded paths
    # Blame windows' tantrums not me
    src_dir = "C:/Users/Asus/Desktop/Jupiter's ballroom of Regret/vision_project/datasets/caltech256/256_ObjectCategories" 
    final_root = "datasets/caltech256_split"
    
    if not os.path.exists(src_dir):
        print(f"Source {src_dir} not found! Check your folder name.")
        return

    all_classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    
    for split in ["train", "test"]:
        os.makedirs(os.path.join(final_root, split), exist_ok=True)

    print(f"Splitting {len(all_classes)} classes...")
    for cls in tqdm(all_classes):
        cls_path = os.path.join(src_dir, cls)
        images = sorted(os.listdir(cls_path))
        
        # 80/20 split
        split_idx = int(len(images) * 0.8)
        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        for split, imgs in [("train", train_imgs), ("test", test_imgs)]:
            dst_dir = os.path.join(final_root, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for img in imgs:
                shutil.copy(os.path.join(cls_path, img), os.path.join(dst_dir, img))

    print("\nDone. You can now delete the original '256_ObjectCategories' to save space.")

if __name__ == "__main__":
    physical_split()