from __future__ import division
import numpy as np
import printer


# Error function which print message and exit
def error(msg):
    print(msg)
    exit(1)


# Model function
def model(X, theta):
    return X.dot(theta)


# Cost function
def cost_function(X, y, theta):
    m = len(y)

    return 1 / m * float(np.sum((model(X, theta) - y) ** 2))


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
    try:
        data = np.genfromtxt('../data.csv', delimiter=',')[1:]
    except:
        error("\033[31mImpossible to read data file.\033[0m")

    x = data[:, 0]
    y = data[:, 1]

    y = y.reshape(y.shape[0], 1)
    x = x.reshape(x.shape[0], 1)

    unormalized_x = x

    x = normalize(x)

    return x, y, unormalized_x


# Return coefficient of determination
def get_determination_coef(y, predictions):
    u = ((y - predictions) ** 2).sum()
    v = ((y - y.mean()) ** 2).sum()
    return 1 - u / v


# Main function
def linear_regression():
    # Read data
    x, y, unormalized_x = parse_data()

    X = np.hstack((x, np.ones(x.shape)))

    theta = np.random.randn(2, 1)

    theta_final, cost_history = gradient_descent(X, y, theta, 0.1, 1000)
    predictions = model(X, theta_final)

    coef = get_determination_coef(y, predictions)
    print("Coefficient of determination : " + str(coef))
    printer.print_plots(x, y, X, cost_history, predictions, unormalized_x)

    try:
        file = open("../tmp/save.txt", "w")
        file.write(str(float(theta_final[0])) + "," + str(float(theta_final[1])))
        file.close()
    except:
        error("\033[31mError during writing theta value.\n\033[0;0m")
    return theta_final


if __name__ == "__main__":
    linear_regression()
