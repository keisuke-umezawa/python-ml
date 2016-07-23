import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from sklearn import linear_model, preprocessing

def main():
    # data
    data = scio.loadmat('ex5data1.mat')
    X = data['X']
    Xval = data['Xval']
    y = data['y']
    yval = data['yval']
    ytest = data['ytest']
    
    # model
    model = linear_model.Ridge(alpha = 0.0)
    model.fit(X, y)

    px = np.array(np.linspace(np.min(X), np.max(X), 100)).reshape(-1, 1)
    py = model.predict(px)
    plt.plot(px, py)
    plt.scatter(X, y)
    plt.show()

    # plot learning curve
    error_train = np.zeros(11)
    error_val = np.zeros(11)
    model = linear_model.Ridge(alpha = 0.0)
    for i in range(1, 12):
        model.fit(X[0:i], y[0:i])
        error_train[i-1] \
            = 0.5 * sum((y[0:i] - model.predict(X[0:i])) ** 2.0) / i
        error_val[i-1] \
            = 0.5 * sum((yval - model.predict(Xval)) ** 2.0) / yval.size

    px = np.arange(1, 12)
    plt.plot(px, error_train, label="Train")
    plt.plot(px, error_val, label="Cross Validation")
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

    # polynomial
    poly = preprocessing.PolynomialFeatures(degree=8, include_bias=False)
    X_poly = poly.fit_transform(X)

    model = linear_model.Ridge(alpha=0.0)
    model.fit(X_poly, y)
    px = np.array(np.linspace(np.min(X)-10,np.max(X)+10,100)).reshape(-1,1)
    px_poly = poly.fit_transform(px)
    py = model.predict(px_poly)
    plt.plot(px, py)
    plt.scatter(X, y)
    plt.show()
    
    # find lambda
    poly = preprocessing.PolynomialFeatures(degree=8, include_bias=False)
    X_poly = poly.fit_transform(X)
    Xval_poly = poly.fit_transform(Xval)

    error_train = np.zeros(9)
    error_val = np.zeros(9)
    lambda_values \
        = np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    for i in range(0, lambda_values.size):
        model = linear_model.Ridge(alpha = lambda_values[i]/10, normalize=True)
        model.fit(X_poly, y)
        error_train[i] \
            = 0.5 * sum((y - model.predict(X_poly)) ** 2.0) / y.size
        error_val[i] \
            = 0.5 * sum((yval - model.predict(Xval_poly)) ** 2.0) / yval.size


    px = lambda_values
    plt.plot(px, error_train, label="Train")
    plt.plot(px, error_val, label="Cross Validation")
    plt.xlabel("Lambda")
    plt.ylabel("Error")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    sys.exit(int(main() or 0))