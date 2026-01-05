from data import DataLoader, Preprocessor
import matplotlib.pyplot as plt
import torch
print("GPU is active:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10
X, y = DataLoader.load_cifar10()

# Take first 1000 samples
mask = range(1000)
X_train = X[mask]
y_train = y[mask]

# this is really bad code, fix it later 
def plot(X_train, y_train):
    plt.figure(figsize=(15, 3))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(X_train[i], cmap= 'gray')
        plt.title(y_train[i])
        plt.axis('off')

    plt.show(block=True)  

# Everything is working fine till here (I actually printed the examples as numbers and checked)

X_train = Preprocessor.luminance(X_train)

print(X_train[0,:,:,0])

