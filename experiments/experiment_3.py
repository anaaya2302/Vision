from src.data import Preprocessor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import itertools
import os
from skimage.metrics import structural_similarity as ssim
import csv


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_PATHS = [
    os.path.join(BASE_DIR, "assets", "doggie.jpeg"),
    os.path.join(BASE_DIR, "assets", "me at the beach.jpeg"),
    os.path.join(BASE_DIR, "assets", "my red hair.jpeg"),
    os.path.join(BASE_DIR, "assets", "bird.jpeg"),
    os.path.join(BASE_DIR, "assets", "book.jpeg"),
    os.path.join(BASE_DIR, "assets", "charminar.jpeg"),
    os.path.join(BASE_DIR, "assets", "fire.jpeg"),
    os.path.join(BASE_DIR, "assets", "flower.jpeg"),
    os.path.join(BASE_DIR, "assets", "footsteps.jpeg"),
    os.path.join(BASE_DIR, "assets", "food.jpeg"),
    os.path.join(BASE_DIR, "assets", "go karting.jpeg"),
    os.path.join(BASE_DIR, "assets", "ice cream.jpeg"),
    os.path.join(BASE_DIR, "assets", "palm.jpeg"),
    os.path.join(BASE_DIR, "assets", "path.jpeg"),
    os.path.join(BASE_DIR, "assets", "plane wing.jpeg"),
    os.path.join(BASE_DIR, "assets", "redbull.jpeg"),
    os.path.join(BASE_DIR, "assets", "spain castle.jpeg"),
    os.path.join(BASE_DIR, "assets", "statue of liberty.jpeg"),
    os.path.join(BASE_DIR, "assets", "street in spain.jpeg"),
    os.path.join(BASE_DIR, "assets", "sunkissed doggie.jpeg"),
    os.path.join(BASE_DIR, "assets", "times square.jpeg")
    
]


def reconstruct(best_match, edge_templates):
    templates_np = edge_templates[:, 0].cpu().detach().numpy()  # 9, 3, 3
    indices = best_match[0, 0].cpu().detach().numpy()           # H//3, W//3
    reconstructed = templates_np[indices]                        # H//3, W//3, 3, 3
    H, W = indices.shape
    reconstructed = reconstructed.transpose(0, 2, 1, 3).reshape(H * 3, W * 3)
    return reconstructed


def grid_search(image, device, image_name="image"):

    alphas     = [5.0, 10.0, 20.0]
    rs         = [0.1, 0.2, 0.3, 0.4]
    thresholds = [0.2, 0.3, 0.4, 0.5]

    out_dir = os.path.join(BASE_DIR, "experiments", "grid_search", image_name)
    os.makedirs(out_dir, exist_ok=True)

    edge_templates = Preprocessor.get_edge_templates(device)



    results = []


    for alpha, r, threshold in itertools.product(alphas, rs, thresholds):

        image_t       = image.unsqueeze(0)
        image_t       = Preprocessor.luminance(image_t, channels_last=True)
        edge_grads    = Preprocessor.edge_grads(image_t)
        _, edge_probs = Preprocessor.edge_probs(*edge_grads, alpha=alpha, r=r)
        edge_binary   = Preprocessor.thresholding_grayscale(edge_probs, threshold=threshold)


        best_match_binary = Preprocessor.template_match(edge_binary.float(), edge_templates)

        recon_binary = reconstruct(best_match_binary, edge_templates)

        h, w = recon_binary.shape
     
        edge_probs = edge_probs[0,0].cpu().detach().numpy()

        edge_probs = edge_probs[:h, :w]


        ssim_binary = ssim(edge_probs, recon_binary, data_range=1.0)

        results.append((alpha, r, threshold, ssim_binary))
        print(f"[{image_name}] alpha={alpha:5.1f}  r={r:.1f}  threshold={threshold:.1f}"
              f"  → SSIM_binary={ssim_binary:.4f}")



    best = max(results, key=lambda x: x[3])
    print(f"\n[{image_name}] BEST (binary): "
          f"alpha={best[0]} threshold={best[1]}  "
          f"SSIM={best[3]:.4f}\n")

    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "r", "threshold", "ssim_prob", "ssim_binary"])
        writer.writerows(results)

    return results


def main():
    print("GPU is active:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_results = {}

    for path in IMAGE_PATHS:
        assert os.path.exists(path), f"Image not found: {path}"
        image_name = os.path.splitext(os.path.basename(path))[0]

        image = Image.open(path).convert("RGB")
        image = torch.from_numpy(np.array(image)).to(device)

        results = grid_search(image, device, image_name=image_name)
        all_results[image_name] = results

    print("\n===== SUMMARY: best SSIM_binary per image =====")
    for name, results in all_results.items():
        best = max(results, key=lambda x: x[3])
        print(f"{name:>20}  alpha={best[0]}  r={best[1]}  "
              f"threshold={best[2]}  SSIM={best[3]:.4f}")


if __name__ == "__main__":
    main()
