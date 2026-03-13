import numpy as np 
import torch as torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10


class DataLoader:
    


    def load_cifar10(train=True):
        dataset = CIFAR10(
            root="../datasets",
            train=train,
            download=True
    )

        X = np.stack([np.array(img) for img, _ in dataset])
        y = np.array([label for _, label in dataset])

        return X, y

class Preprocessor:
    
     

     EDGE_TEMPLATES = torch.zeros(9, 1, 3, 3)

     # 1. Diagonal left-to-right (top-left to bottom-right)
     EDGE_TEMPLATES[0, 0] = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=torch.float32)

     # 2. Diagonal right-to-left (top-right to bottom-left)
     EDGE_TEMPLATES[1, 0] = torch.tensor([
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]
        ], dtype=torch.float32)

     # 3. Upper row (horizontal, top)
     EDGE_TEMPLATES[2, 0] = torch.tensor([
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=torch.float32)

     # 4. Middle row (horizontal, middle)
     EDGE_TEMPLATES[3, 0] = torch.tensor([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0]
        ], dtype=torch.float32)

     # 5. Lower row (horizontal, bottom)
     EDGE_TEMPLATES[4, 0] = torch.tensor([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1]
        ], dtype=torch.float32)

     # 6. Left column (vertical, left)
     EDGE_TEMPLATES[5, 0] = torch.tensor([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0]
        ], dtype=torch.float32)

     # 7. Middle column (vertical, center)
     EDGE_TEMPLATES[6, 0] = torch.tensor([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0]
        ], dtype=torch.float32)

     # 8. Right column (vertical, right)
     EDGE_TEMPLATES[7, 0] = torch.tensor([
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ], dtype=torch.float32)

     # 9. No edge
     EDGE_TEMPLATES[8, 0] = torch.zeros(3, 3, dtype=torch.float32)
            
     def luminance(
        X, 
        luminance = (0.299, 0.587, 0.114),
        normalised=True,
        channels_last = True
    ):
        """
        Convert image to grayscale with luminance weights

        Inputs:
            - X: Images of dim N, H, W, 3 [numpy array / torch tensor]
            OR Images of dim N, 3, H, W [numpy array / tensor ] with channels_last = False
            - luminance: List/tuple containing weight of each color channel in order (R,G,B)

        Output:
            - Normalized images in grayscale [torch tensor]
        """
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()

       
        if channels_last:
            X = X.permute(0, 3, 1, 2) 

        luminance = torch.as_tensor(luminance, dtype = torch.float32, device=X.device)
      
        luminance = luminance.view(1, 3, 1, 1) # Ensures broadcasting along the correct channel

        
        X = torch.sum(X * luminance, dim=1, keepdim=True)
 
        if normalised:
           X = X.float()/ 255.0

        return X
     
     def thresholding_grayscale(X, threshold=0.5):
        """
        Determine if pixels are 'on' or 'off'

        Inputs:
            - X: Grayscale images of shape N, 1, H, W [numpy array / torch tensor]
            - threshold: intensities above threshold will be True (1) 
        
        Output:
            - X as 'on' and 'off' pixels. on = 1, off = 0
        """
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X)

        X = (X > threshold).byte() #uint 8

        return X
    
     def edge_grads(X):
     # The padding is hapzard for now. myb. will fix
         """
         Calculate edge gradients at each pixel. [horizontal and vertical]

         Inputs:
            - X: Tesnor of shape N, 1, H, W [preferably luminance normalized] PLS DO NOT PLAY FUNNY WITH SHAPES


         Output:
            - G_X: Tensor of shape N, H, W [storing gradient at each pixel along x axis, horizontal edge]
            - G_Y: Tensor of shape N, H, W [storing gradient at each pixel along y axis, vertical edge]
         """
         
         assert X.shape[1] == 1, "Incorrect Input shape"
         assert X.ndim == 4, "Incorrect number of Input dimensions"
            

         x_kernel = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
            ], dtype=torch.float32, device = X.device)

         y_kernel = torch.tensor([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
            ], dtype=torch.float32, device = X.device)
         
         kernels = torch.stack((x_kernel, y_kernel)) # Shape 2, 3, 3
         kernels = kernels.unsqueeze(1) # Shape 2, 1, 3 ,3

         edges =  F.conv2d(X, kernels, padding='same') # Shape N, 2, H, W [stride = 1]

         G_x = edges[:, 0, :, :] # Shape N, H, W
         G_y = edges[:, 1, :, :]



         return G_x, G_y

     def edge_probs(G_x, G_y, alpha=10.0, r=0.2):
     # I should probably keep alpha and r as learnable
     # But out of sheer laziness, they're hyperparams for now
        """
        Using edge gradients, calculate probability of edge using vector sum of gradients along both directions

        Inputs: 
            - alpha: Controls slope of sigmoid, therefore controlling how quickly your probability changes w.r.t edge strength
            - r: Threshold [if low, high edge sensitivity]
            - G_X: Shape N, H, W [gradient at each pixel along x axis, horizontal edge]
            - G_Y: Shape N, H, W [gradient at each pixel along y axis, vertical edge]  

        Outputs:
            - edge_mag: Edge gradient vector at each pixel [Tensor of shape N, H, W, 1]
            - edge_prob: Probability of there being an edge at that pixel [Tensor of shape N, H, W, 1]
        """

        assert G_x.device == G_y.device, "horizontal and vertical gradients are on different devices"

        alpha = torch.nn.Parameter(torch.tensor(alpha, device = G_x.device))
        r = torch.nn.Parameter(torch.tensor(r, device = G_x.device))        
        edge_mag = torch.sqrt(G_x ** 2 + G_y ** 2 + 1e-6).unsqueeze(1) # shape N, H, W, 1



        p_edge = torch.sigmoid(alpha * (edge_mag-r))

        
        return edge_mag, p_edge
     
     def edge_angles(G_x, G_y):
        """
        Using edge gradients and arctan, calculate the edge angle

        Inputs:
            - G_X: Shape N, H, W [gradient at each pixel along x axis, horizontal edge]
            - G_Y: Shape N, H, W [gradient at each pixel along y axis, vertical edge] 
        Output: 
            - edge_angle: The angle of the edge at each pixel in rads [Shape N, H, W, 1]
        """

        
        edge_angles = torch.arctan2(G_y, G_x)

        edge_angles = edge_angles.unsqueeze(1)

        return edge_angles

     def template_match(X, edge_templates):
         scores = F.conv2d(X, edge_templates)
         best_match = torch.argmax(scores, dim=1, keepdim=True)
         return best_match
          
