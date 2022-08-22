import matplotlib.pyplot as plt
import torch  # pytorch API useful for deep learning implementations
import numpy as np  # numpy API for arithmetics


def convert_numpy_array_to_pytorch(X):
    X = torch.from_numpy(X)
    return X


def create_data():
    X = np.arange(-8, 8, 0.01)
    X = convert_numpy_array_to_pytorch(X)
    return X


def get_loss_y_0(y_hat):
    return -np.log(1 - y_hat)


def get_loss_y_1(y_hat):
    return -np.log(y_hat)


def get_loss_gen(y, y_hat):
    return (-y) * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)


def plot_losses(y_hat, loss_y_0, loss_y_1):
    plt.plot(y_hat, loss_y_0, label='$\ell(y=0,\hat{y})$')
    plt.plot(y_hat, loss_y_1, label='$\ell(y=1,\hat{y})$')

    plt.xlabel('$\hat{y}$')
    plt.legend()
    plt.title('cross entropy')
    plt.show()
