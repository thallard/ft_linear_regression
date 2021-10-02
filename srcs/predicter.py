import sys
import numpy as np

def get_theta():
    thetas = 0
    try:
        with open("../tmp/save.txt", "r") as file:
            buf = file.read()
            thetas_map = map(float, buf.split(','))
            thetas = list(thetas_map)
    except:
        print("\033[31mError during writing theta value.\n\033[0;0m")

    return thetas


def normalize(x, init):
    return (x - min(init)) / (max(init) - min(init))


def prediction():
    if len(sys.argv) != 2:
        print("\033[31mIncorrect number of parameters.\033[0m")
        exit(1)
    km = 0
    try:
        km = int(sys.argv[1])
    except:
        print("\033[31mIncorrect value given.\033[0m")
        exit(1)
    try:
        data = np.genfromtxt('../data.csv', delimiter=',')[1:]
    except:
        print("\033[31mData file is unreachable.\033[31m\n")
        exit(1)
    thetas = get_theta()
    print(thetas[0], thetas[1], thetas[1] * normalize(km, data[:, 0]))
    print("Estimated price :", round(float(thetas[0] + thetas[1] * normalize(km, data[:, 0])), 1))


if __name__ == "__main__":
    prediction()
