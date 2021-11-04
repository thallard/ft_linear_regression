import numpy as np
from matplotlib import pyplot as plt

import linear_regression


def print_plots(x, y, X, cost_history, predictions, unormalized_x):
    theta_random = np.random.randn(2, 1)

    # Create window
    plt.figure(figsize=(10, 10), dpi=80)

    # Create raw data plot
    plt.subplot(2, 2, 1)
    plt.title("Raw data")
    plt.scatter(unormalized_x, y)
    plt.xlabel('Kilometers')
    plt.ylabel('Price')

    # Create random theta data plot
    plt.subplot(2, 2, 2)
    plt.title("Before Machine Learning")
    plt.scatter(unormalized_x, y)
    plt.xlabel('Kilometers')
    plt.ylabel('Price')
    plt.plot(unormalized_x, linear_regression.model(X, theta_random), c='r')

    # Create linear regression plot
    plt.subplot(2, 2, 3)
    plt.title("After Machine Learning")
    plt.scatter(unormalized_x, y)
    plt.xlabel('Kilometers')
    plt.ylabel('Price')
    plt.plot(unormalized_x, predictions, c='r')

    # Create cost function plot
    plt.subplot(2, 2, 4)
    plt.title("Cost function")
    plt.plot(range(len(cost_history) - 40), cost_history[40:])
    plt.xlabel('Iterations')
    plt.ylabel('Cost price')
    plt.yscale('log')
    plt.tick_params()

    plt.tight_layout()
    plt.show()
