from __future__ import division

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def model(X, theta):
    return X.dot(theta)


def cost_function(X, y, theta):
    m = len(y)

    return 1 / float(2 * m) * float(np.sum((model(X, theta) - y) ** 2))


def grad(X, y, theta):
    m = len(y)
    return 1 / m * X.T.dot(model(X, theta) - y)


# Machine learning function using gradient descent
def gradient_descent(X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)

    for i in range(0, n_iterations):
        theta -= learning_rate * grad(X, y, theta)
        cost_history[i] = cost_function(X, y, theta)

    return theta, cost_history


def normalize(x):
    return (x - min(x)) / (max(x) - min(x))


def normalize2(x, init):
    return (x - min(init)) / (max(init) - min(init))


# Using Pandas, read each information in .csv and create two NumPy arrays
def parse_data():
    data = np.genfromtxt('../data.csv', delimiter=',')[1:]
    x = data[:, 0]
    y = data[:, 1]

    y = y.reshape(y.shape[0], 1)
    x = x.reshape(x.shape[0], 1)

    x = normalize(x)

    return x, y


def coef_deter(y, predictions):
    u = ((y - predictions) ** 2).sum()
    v = ((y - y.mean()) ** 2).sum()
    return 1 - u / v


# Main function
def main():
    data = np.genfromtxt('../data.csv', delimiter=',')[1:]
    x, y = parse_data()

    X = np.hstack((x, np.ones(x.shape)))

    theta = np.random.randn(2, 1)
    theta_random = np.random.randn(2, 1)

    theta_final, cost_history = gradient_descent(X, y, theta, 0.1, 1000)
    predictions = model(X, theta_final)

    print(theta_final)
    print(theta[1] + theta[0] * normalize2(240000, data[:, 0]))

    plt.subplot(2, 2, 1)
    plt.title("Raw data")
    plt.scatter(x, y)
    plt.subplot(2, 2, 2)
    plt.title("Before Machine Learning")
    plt.scatter(x, y)
    plt.plot(x, model(X, theta_random), c='r')
    plt.tight_layout(pad=2.0)
    plt.subplot(2, 2, 3)
    plt.title("After Machine Learning")
    plt.scatter(x, y)
    plt.plot(x, predictions, c='r')
    plt.subplot(2, 2, 4)
    plt.title("Cost function")
    plt.plot(range(1000), cost_history)
    plt.tick_params()
    plt.show()
    print(cost_history)

    print(coef_deter(y, predictions))

if __name__ == "__main__":
    main()
