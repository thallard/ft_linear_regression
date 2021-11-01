import numpy as np
from matplotlib import pyplot as plt

import linear_regression


def print_plots(x, y, X, cost_history, predictions):
    theta_random = np.random.randn(2, 1)

    plt.subplot(2, 2, 1)
    plt.title("Raw data")
    plt.scatter(x, y)
    plt.subplot(2, 2, 2)
    plt.title("Before Machine Learning")
    plt.scatter(x, y)
    plt.plot(x, linear_regression.model(X, theta_random), c='r')
    plt.subplot(2, 2, 3)
    plt.title("After Machine Learning")
    plt.scatter(x, y)
    plt.plot(x, predictions, c='r')
    plt.subplot(2, 2, 4)
    plt.title("Cost function")
    plt.plot(range(len(cost_history)), cost_history)
    plt.tick_params()
    plt.show()
