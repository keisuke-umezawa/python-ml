import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

def main():
    # data
    data = pd.read_csv("ex2data1.txt", header=None)
    X = np.array([data[0], data[1]]).T
    y = np.array(data[2])

    # plot 
    pos = (y == 1)
    neg = (y == 0)
    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='b')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', c='y')
    plt.legend(['Admitted', 'Not Admitted'], scatterpoints=1)
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")

    # logistic regression
    model = linear_model.LogisticRegression(C = 1000000.0)
    model.fit(X, y)

    # extract model parameter (theta0, theta1, theta2)
    theta0 = model.intercept_
    [[theta1, theta2]] = model.coef_

    # plot decision boundary
    plot_x = np.array([min(X[:, 0]) - 2, max(X[:, 0]) + 2])
    plot_y = - (theta0 + theta1 * plot_x) / theta2
    plt.plot(plot_x, plot_y, 'b')

    plt.show()

if __name__ == "__main__":
    sys.exit(int(main() or 0))