import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn import linear_model

def main():
    # data
    data = scio.loadmat('ex3data1.mat')
    X = data['X'] # 5000x400
    y = data['y'].ravel() # matrix -> vector

    # logistic regression
    model = linear_model.LogisticRegression(
        penalty = 'l2', C = 10.0)
    model.fit(X, y) # learning
    score = model.score(X, y) # answer
    print(score)

    # output plot
    wrong_index = np.array(np.nonzero(np.array(
        [model.predict(X) != y]).ravel())).ravel()
    wrong_sample_index = np.random.randint(0, len(wrong_index), 25)
    fig = plt.figure()
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, 
        wspace=0.5, hspace=0.5)
    for i in range(0, 25):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.axis('off')
        ax.imshow
        ax.imshow(X[wrong_index[wrong_sample_index[i]]].reshape(20,20).T, cmap = plt.get_cmap('gray'))
        ax.set_title(str(model.predict(X[wrong_index[wrong_sample_index[i]]])[0]))
    plt.show()


if __name__ == "__main__":
    sys.exit(int(main() or 0))