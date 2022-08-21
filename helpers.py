import torch # pytorch API useful for deep learning implementations
import numpy as np # numpy API for arithmetics


def convert_numpy_array_to_pytorch(X):
    X = torch.from_numpy(X)
    return X


def create_data():
    X = np.arange(-8, 8, 0.01)
    X = convert_numpy_array_to_pytorch(X)
    return X
