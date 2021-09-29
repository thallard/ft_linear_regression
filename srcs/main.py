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


# Using Pandas, read each information in .csv and create two NumPy arrays
def parse_data():
    df = pd.read_csv('../data.csv')
    df.head()
    df.astype('float64')
    kms = df.to_numpy()

    temp_x, temp_y = kms.T

    # Cast my temp Vector2 as float64
    temp_x = temp_x.astype(np.float64)
    temp_y = temp_y.astype(np.float64)

    temp_y = temp_y.reshape(temp_y.shape[0], 1)
    temp_x = temp_x.reshape(temp_x.shape[0], 1)
    x = temp_x / 10000
    y = temp_y / 1000
    return x, y


# Main function
def main():
    x, y = parse_data()

    X = np.hstack((x, np.ones(x.shape)))

    theta = np.random.randn(2, 1)

    n_iterations = 5000
    learning_rate = 0.01

    theta_final, cost_history = gradient_descent(X, y, theta, learning_rate, n_iterations)
    predictions = model(X, theta_final)
    plt.scatter(x, y)
    plt.plot(x, predictions, c='r')
    plt.show()


if __name__ == "__main__":
    main()
