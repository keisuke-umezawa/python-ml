import sys
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn import decomposition

def main():
    # image
    data = scio.loadmat('ex7faces.mat')
    X = data['X'] # x is 5000x1024

    # fig
    fig = plt.figure()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    for i in range(0, 100):
        ax = fig.add_subplot(10, 10, i + 1)
        ax.axis('off')
        ax.imshow(X[i].reshape(32, 32).T, cmap = plt.get_cmap('gray'))

    plt.show()

    # pca
    pca = decomposition.PCA(n_components=100)
    pca.fit(X)

    fig = plt.figure()
    for i in range(0,36):
        ax = fig.add_subplot(6,6,i+1)
        ax.axis('off')
        ax.imshow(pca.components_[i].reshape(32,32).T, cmap = plt.get_cmap('gray'))

    plt.show()


if __name__ == "__main__":
    sys.exit(int(main() or 0))