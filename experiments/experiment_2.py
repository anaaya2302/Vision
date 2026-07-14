from src.data import Preprocessor
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch as torch
import itertools
import os


def properties(image):
   
    image =  image.unsqueeze(0) #1, H, W, 3
    image = Preprocessor.luminance(image, channels_last=True)
    edge_grads = Preprocessor.edge_grads(image)
    _, edge_probs = Preprocessor.edge_probs(*edge_grads)
    edge_binary = Preprocessor.thresholding_grayscale(edge_probs)
    return edge_probs, edge_binary

def binary_map(edge_binary, edge_templates):
    edge_binary = edge_binary.float()
    best_match = Preprocessor.template_match(edge_binary, edge_templates)
    return best_match

def prob_map(edge_probs, edge_templates):
    best_match = Preprocessor.template_match(edge_probs, edge_templates)
    return best_match

def reconstruct(best_match, edge_templates):
    templates_np = edge_templates[:, 0].cpu().detach().numpy()  # 9, 3, 3
    indices = best_match[0, 0].cpu().detach().numpy()  # H//3, W//3
    # index directly into templates, then reshape into image
    reconstructed = templates_np[indices]  # H//3, W//3, 3, 3
    H, W = indices.shape
    reconstructed = reconstructed.transpose(0, 2, 1, 3).reshape(H*3, W*3)
    return reconstructed

def plot(image, best_match_prob, best_match_binary, edge_templates):
    """
    Inputs:
        - image: Original numpy array H, W, 3
        - best_match_prob: Tensor of shape 1, 1, H, W (template indices from soft probs)
        - best_match_binary: Tensor of shape 1, 1, H, W (template indices from binary map)
        - edge_templates: Tensor of shape 9, 1, 3, 3
    """



    recon_prob   = reconstruct(best_match_prob, edge_templates)
    recon_binary = reconstruct(best_match_binary, edge_templates)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Soft probability templates")
    plt.imshow(recon_prob, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Binary threshold templates")
    plt.imshow(recon_binary, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()



def grid_search(image, device):
    
    alphas     = [5.0, 10.0, 20.0]
    rs         = [0.1, 0.2, 0.3, 0.4]
    thresholds = [0.2, 0.3, 0.4, 0.5]

    os.makedirs("grid_search", exist_ok=True)

    edge_templates = Preprocessor.get_edge_templates(device)

    for alpha, r, threshold in itertools.product(alphas, rs, thresholds):
        
        image_t = image.unsqueeze(0)
        image_t = Preprocessor.luminance(image_t, channels_last=True)
        edge_grads = Preprocessor.edge_grads(image_t)
        _, edge_probs = Preprocessor.edge_probs(*edge_grads, alpha=alpha, r=r)
        edge_binary = Preprocessor.thresholding_grayscale(edge_probs, threshold=threshold)

        best_match_prob    = Preprocessor.template_match(edge_probs, edge_templates)
        best_match_binary  = Preprocessor.template_match(edge_binary.float(), edge_templates)

        recon_prob   = reconstruct(best_match_prob, edge_templates)
        recon_binary = reconstruct(best_match_binary, edge_templates)

        h, w = image.cpu().numpy().shape[:2]
        fig, axes = plt.subplots(1, 3, figsize=(w*3/100, h/100))

        fig.suptitle(f"alpha={alpha}  r={r}  threshold={threshold}", fontsize=13)

        axes[0].imshow(image.cpu().numpy())
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(recon_prob, cmap="gray")
        axes[1].set_title("Soft probability")
        axes[1].axis("off")

        axes[2].imshow(recon_binary, cmap="gray")
        axes[2].set_title("Binary threshold")
        axes[2].axis("off")

        plt.tight_layout()
        fname = f"grid_search/a{alpha}_r{r}_t{threshold}.png"
        plt.savefig(fname)
        plt.close()
        print(f"saved {fname}")
        
def main():
    print("GPU is active", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert len(sys.argv) == 2, "invalid number of arguments"
    assert sys.argv[1].lower().endswith(("jpeg", "jpg")), "invalid file format"


    image = Image.open(sys.argv[1])
    image = np.array(image)
    image = torch.from_numpy(image) #H,w,3
    image = image.to(device)

    grid_search(image, device)


if __name__ == "__main__":
    main()