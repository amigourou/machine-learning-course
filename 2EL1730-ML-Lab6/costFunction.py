from numpy import *
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params

def costFunction(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the cost function of the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor
    
    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Unroll Params
    Theta = roll_params(nn_weights, layers)
    
    # You need to return the following variables correctly 
    J = 0;
    
    # ================================ TODO ================================
    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a 
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = zeros((num_labels, m))
  

    # ================================ TODO ================================
    # In this point calculate the cost of the neural network (feedforward)
       

    return J

    

