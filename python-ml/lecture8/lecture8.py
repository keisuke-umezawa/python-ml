import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def main():
    # image
    img = np.array(Image.open('bird_small.png'))
    img_flat = img.reshape(128*128, 3) / 255

    # model
    model = KMeans(n_clusters=16).fit(img_flat)

    img_comp = model.cluster_centers_[model.labels_].reshape(128, 128, 3)
    plt.imshow(img_comp)
    plt.show()

if __name__ == "__main__":
    sys.exit(int(main() or 0))