from numpy import *

def randInitializeWeights(layers):

    num_of_layers = len(layers)

    Theta = []
    for i in range(num_of_layers-1):
        W = zeros((layers[i+1], layers[i] + 1),dtype = 'float64')
        #% ====================== YOUR CODE HERE ======================
        # Instructions: Initialize W randomly so that we break the symmetry while
        #               training the neural network.
        #
        # Note: The first row of W corresponds to the parameters for the bias units
        #
        epsilon = 0.12
        W = 2.0*epsilon*random.rand(layers[i+1], layers[i] + 1) - epsilon
        Theta.append(W)
                
    return Theta
            
