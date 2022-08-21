import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
import sys
import torch.nn as nn


def logistic(Z):
    return 1 / (1+torch.exp( -Z ))


def create_data():
    x = np.arange(-8,8,0.01)
    print('those are the new data points: ',x)
    return x

def convert_numpy_array_to_pytorch(x):
    x = torch.from_numpy(x)
    return x


def calculate_y_yat(x):
    y_hat = logistic(x)
    return y_hat


def plot_graph(x,y_hat):
    plt.title("Sigmoid activation function")
    plt.plot(x,y_hat)
    plt.show()


if __name__ == '__main__':
    data = create_data()
    data = convert_numpy_array_to_pytorch(data)
    y_hat = calculate_y_yat(data)
    plot_graph(data,y_hat)


