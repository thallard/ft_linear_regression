from __future__ import division

import numpy as np
import printer


# Model function
def model(X, theta):
    return X.dot(theta)


# Cost function
def cost_function(X, y, theta):
    m = len(y)

    return 1 / float(2 * m) * float(np.sum((model(X, theta) - y) ** 2))


# Gradient descent
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


# Normalize data set to a 0 to 1 interval
def normalize(x):
    return (x - min(x)) / (max(x) - min(x))


# Using Numpy, read each information in .csv and create two arrays
def parse_data():
    data = None
    try:
        data = np.genfromtxt('../data.csv', delimiter=',')[1:]
    except:
        print("\033[31mImpossible to read data file.\033[0m")
        exit(1)

    x = data[:, 0]
    y = data[:, 1]

    y = y.reshape(y.shape[0], 1)
    x = x.reshape(x.shape[0], 1)

    x = normalize(x)

    return x, y


# Return coefficient of determination
def get_determination_coef(y, predictions):
    u = ((y - predictions) ** 2).sum()
    v = ((y - y.mean()) ** 2).sum()
    return 1 - u / v


# Main function
def linear_regression():
    # Read data
    x, y = parse_data()

    X = np.hstack((x, np.ones(x.shape)))

    theta = np.random.randn(2, 1)

    theta_final, cost_history = gradient_descent(X, y, theta, 0.1, 1000)
    predictions = model(X, theta_final)

    printer.print_plots(x, y, X, cost_history, predictions)
    coef = get_determination_coef(y, predictions)
    print("Coefficient of determination : " + str(coef))

    try:
        file = open("../tmp/save.txt", "w")
        file.write(str(float(theta_final[0])) + "," + str(float(theta_final[1])))
        file.close()
    except:
        print("\033[31mError during writing theta value.\n\033[0;0m")
    return theta_final


if __name__ == "__main__":
    linear_regression()
