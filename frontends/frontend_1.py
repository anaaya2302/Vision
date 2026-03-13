from src.data import Preprocessor
import sys
import numpy as np
from PIL import Image
from matplotlib.colors import hsv_to_rgb
import torch as torch
import matplotlib.pyplot as plt


# Underlying functions are written in torch
# Back and forth between torch and numpy is technically computationally inefficient
# But for design and matplotlib, 
# I've made all frontend functions take input and give output in numpy arrays
# With functions like from_numpy, there's no copying in memory
# Realistically the only trade off ends up being taking memory from CPU to GPU and back
# That is also optional

def properties(image, device="cpu"):
    """
    Input: 
        - image: Numpy array of shape H, W, 3
        - device: use cuda for efficiency, cpu by default
    Outputs:
        - edge_probs: Edge probabilities for each pixel of image [Numpy array of shape 1, H, w, 1]
        - edge_angles: Angle of edge at each pixel in rads [Numpy array of shape 1, H, w, 1]
    """
    image = torch.from_numpy(image)
    image = image.to(device)

    image = image.unsqueeze(0)

    image = Preprocessor.luminance(image)

    edge_grads = Preprocessor.edge_grads(image)

    _, edge_probs = Preprocessor.edge_probs(*edge_grads)

    edge_angles = Preprocessor.edge_angles(*edge_grads)

    edge_probs = edge_probs.cpu().detach().numpy()

    edge_angles = edge_angles.cpu().detach().numpy()

    image = image.cpu().detach().numpy()

    return edge_probs, edge_angles



def plot(image, edge_probs, edge_angles):

    """
    Inputs: 
        -image: Numpy array of image to plot
        - edge_probs: Numpy array of edge probabilities for each pixel of image [shape 1, H, w, 1]
        - edge_angles: Numpy array of angle of edge at each pixel in rads [shape 1, H, w, 1]
    
    Output:
        - N/A [plots image beside it's edge probabilities and angles using HSV color code]
    """

    edge_angles = edge_angles[0, 0, :, :]   # (H, W)
    edge_probs  = edge_probs[0, 0, :, :]    # (H, W)

    hue = (edge_angles + np.pi) / (2 * np.pi)


    H, W = hue.shape

    hsv = np.zeros((H, W, 3), dtype=np.float32)
    hsv[..., 0] = hue              # H
    hsv[..., 1] = 1.0              # S
    hsv[..., 2] = edge_probs      # V

    rgb_edges = hsv_to_rgb(hsv)    # (H, W, 3)



    plt.figure(figsize=(18, 6))

    # --- Original RGB image ---
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis("off")

    # --- Edge angle (hue) + probability (value) ---
    plt.subplot(1, 3, 2)
    plt.title("Edge Angle (Hue) × Probability (Brightness)")
    plt.imshow(rgb_edges)
    plt.axis("off")

    # --- Edge probability alone ---
    plt.subplot(1, 3, 3)
    plt.title("Edge Probability")
    plt.imshow(edge_probs, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")

    plt.tight_layout()
    plt.show()



def main():
    print("GPU is active", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert len(sys.argv) == 2, "invalid number of arguments"
    assert sys.argv[1].lower().endswith(("jpeg", "jpg")), "invalid file format"


    image = Image.open(sys.argv[1])
    image = np.array(image)

    edge_probs, edge_angles = properties(image, device)

    plot(image, edge_probs, edge_angles)
    

# (1, H, W, 1)
    # shape H, W, 3

if __name__ == "__main__":
    main()



