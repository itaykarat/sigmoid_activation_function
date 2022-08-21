# implementation of preceptron
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sys
import torch.nn as nn


"""
    Note: We must have a function that we can calculate it's derivative, it means a STEP function is not allowed.
    
    We need to find an optimal activation function that can let us use gradient calculation - why?
    
    One of the steps in learning the neuron's inputs weight is by gradient descent.
    
    A good solution for the activation function problem is the sigmoid function.
    
    Sigmoid(z) = 1/(1+e^(-z))
    where z is the linear function.
    
    The sigmoid function maps the real line onto [0,1]
    
    We use probabilistic interpretation - likelihood function of our target
    
    y_hat = P(Y=1 |x) 
    
    if y_hat == 0.7 ---- > that means that the input detected as class 1 by 70% liklihood.
    
    
    
    
    

"""

class perceptron(torch.nn.Module):  # ineritance from torch.nn.Module
    def __init__(self, input_dim):
        super(perceptron, self).__init__()  # initialize the inherited class
        self.fc = nn.Linear(input_dim, 1,
                            bias=True)  # Fully connected (fc) ; y= ax+b ; input dim is how many inputs we have
        self.values = torch.tensor([0.0])  # Define the values for the activation function

    # forward function:
    # Input : input to the artificial neuron
    # Output : output of the activation function
    def forward(self, x):
        output = self.fc(x)
        output = torch.heaviside(output, self.values)
        return output


perceptron_obj = perceptron(5)

