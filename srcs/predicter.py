import sys
import numpy as np
from linear_regression import error


# Get thetas from the save.txt
def get_theta():
    thetas = 0
    try:
        with open("../tmp/save.txt", "r") as file:
            buf = file.read()
            thetas_map = map(float, buf.split(','))
            thetas = list(thetas_map)
    except:
        error("\033[31mError during writing theta value.\n\033[0;0m")
    return thetas


# Normalize data set to 0 to 1 interval
def normalize(x, init):
    return (x - min(init)) / (max(init) - min(init))


# Main function, determine the correct price
def prediction():
    if len(sys.argv) != 2:
        error("\033[31mIncorrect number of parameters.\033[0m")
    km = 0
    try:
        km = int(sys.argv[1])
        if km <= 0:
            raise Exception()
    except:
        error("\033[31mIncorrect value given.\033[0m")
    try:
        data = np.genfromtxt('../data.csv', delimiter=',')[1:]
    except:
        error("\033[31mData file is unreachable.\033[31m\n")
    thetas = get_theta()
    price = int(thetas[1] + thetas[0] * normalize(km, data[:, 0]))
    print("Estimated price :", price)


if __name__ == "__main__":
    prediction()
