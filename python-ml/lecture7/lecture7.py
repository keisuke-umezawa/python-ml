import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn import svm

def main():
    # ex1
    data = scio.loadmat('ex6data1.mat')
    X = data['X']
    y = data['y'].ravel()

    pos = (y==1)
    neg = (y==0)
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='y')

    model = svm.SVC(C=100.0, kernel='linear')
    model.fit(X, y)

    px = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    w = model.coef_[0]
    py = - (w[0] * px + model.intercept_[0]) / w[1]
    plt.plot(px, py)
    plt.show()

    # ex2
    data = scio.loadmat('ex6data2.mat')
    X = data['X']
    y = data['y'].ravel()

    pos = (y==1)
    neg = (y==0)
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='y')

    model = svm.SVC(C=1.0, gamma=50.0, kernel='rbf',
                    probability=True)
    model.fit(X, y)

    px = np.arange(0, 1, 0.01)
    py = np.arange(0, 1, 0.01)
    PX, PY = np.meshgrid(px, py)
    XX = np.c_[PX.ravel(), PY.ravel()]
    Z = model.predict_proba(XX)[:, 1]
    Z = Z.reshape(PX.shape)
    plt.contour(PX, PY, Z, levels=[0.5], loinewidths=3)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.show()


if __name__ == "__main__":
    sys.exit(int(main() or 0))