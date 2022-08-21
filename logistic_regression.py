
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
   
   
   
Mathematical notations: 

    Sigmoid(z) = 1/(1+e^(-z))
    where z is a linear function from the form y = ax+b
    
    The sigmoid function maps the real line onto [0,1]
    sig(x) : (-∞, +∞) --> (0, +1)
    
    

probabilistic interpretation: likelihood function of our target

    Let's assume we have a binary classification problem,    
    X = {X1,...Xn} ; will be our data set
    Y = {class_0,class_1} ; will be the discrete classes of the problem
    Xi ∈ Y <===> Xi ∈ {1,0}


    sig(Xi) can represent the probability of that Xi to be clustered as class_1
    
    y_hat = P(Y=1 |x)
    if y_hat == 0.7 ---- > that means that the input detected as class 1 by 70% liklihood.

"""

import torch # pytorch API useful for deep learning implementations
import matplotlib.pyplot as plt # matplotlib API for visualization using plots


# ---------------------------- sigmoid class ----------------------------


class sigmoid:
    def __init__(self, Data_points):
        self.Data_points = Data_points

    def logistic(self, Data_points):
        return 1 / (1 + torch.exp(-Data_points))

    def calculate_y_yat(self):
        y_hat = self.logistic(self.Data_points)
        return y_hat

    def plot_graph(self,y_hat):
        plt.title("Sigmoid activation function")
        plt.plot(self.Data_points, y_hat)
        plt.show()





