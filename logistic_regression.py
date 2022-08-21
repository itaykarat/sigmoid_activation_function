import torch
import numpy as np
import matplotlib.pyplot as plt


""""
Motivation:

   * The sigmoid function is used as an activation function in neural networks
    
    
Properties of the sigmoid function:

   * Domain: (-∞, +∞)
   * Range: (0, +1)
   * σ(0) = 0.5
   * The function is monotonically increasing.
   * The function is continuous everywhere.
   * The function is differentiable everywhere in its domain.

"""




# ---------------------------- HELPERS ----------------------------

def convert_numpy_array_to_pytorch(x):
    x = torch.from_numpy(x)
    return x


def create_data():
    x = np.arange(-8, 8, 0.01)
    x = convert_numpy_array_to_pytorch(x)
    return x


# ---------------------------- sigmoid class ----------------------------


class sigmoid:
    def __init__(self, data):
        self.data = data

    def logistic(self, Z):
        return 1 / (1 + torch.exp(-Z))

    def calculate_y_yat(self, X):
        y_hat = self.logistic(X)
        return y_hat

    def plot_graph(self, x, y_hat):
        plt.title("Sigmoid activation function")
        plt.plot(self, x, y_hat)
        plt.show()


if __name__ == '__main__':
    data = create_data()
    sigmoid_obj = sigmoid(data=data)
    y_hat = sigmoid_obj.calculate_y_yat(data)
    sigmoid_obj.plot_graph(data, y_hat)
