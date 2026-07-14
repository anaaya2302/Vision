from src.data import Preprocessor
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch as torch



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

def plot(image, best_match_binary, edge_templates):
    """
    Inputs:
        - image: Original numpy array H, W, 3
        - best_match_prob: Tensor of shape 1, 1, H, W (template indices from soft probs)
        - best_match_binary: Tensor of shape 1, 1, H, W (template indices from binary map)
        - edge_templates: Tensor of shape 9, 1, 3, 3
    """

    recon_binary = reconstruct(best_match_binary, edge_templates)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(recon_binary, cmap="gray")
    axes[1].set_title("Discrete Edge Reconstruction", fontsize=14)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(f"assets/reconstruction.png", dpi=150, bbox_inches='tight')
    plt.show()



        
def main():
    print("GPU is active", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert len(sys.argv) == 2, "invalid number of arguments"
    assert sys.argv[1].lower().endswith(("jpeg", "jpg")), "invalid file format"


    image = Image.open(sys.argv[1])
    image = np.array(image)
    image = torch.from_numpy(image) #H,w,3
    image = image.to(device)
    edge_templates = Preprocessor.get_edge_templates(device)
    edge_probs, edge_binary = properties(image)
 
    best_binaries = binary_map(edge_binary, edge_templates)
    

    plot(image.cpu().detach().numpy(), best_binaries, edge_templates)

    


if __name__ == "__main__":
    main()