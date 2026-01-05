from src.data import DataLoader, Preprocessor
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

X, y  = DataLoader.load_cifar10()
mask = range(1000)
X_train = X[mask]
y_train = y[mask]

def plot(X_train, y_train):
    plt.figure(figsize=(15, 3))
    for i in range(1):
        plt.subplot(1, 10, i+1)
        plt.imshow(X_train[i], cmap= 'gray')
        plt.title(y_train[i])
        plt.axis('off')

    plt.show(block=True)  



X_train = Preprocessor.luminance(X_train)

edge_prob = Preprocessor.edge_prob(X_train)
plot(X_train, y_train)
print(edge_prob.shape)
print(edge_prob[0, :, :, 0])