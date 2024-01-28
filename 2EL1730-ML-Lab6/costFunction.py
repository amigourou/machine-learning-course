from numpy import *
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params

def costFunction(nn_weights, layers, X, y, num_labels, l):

    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Unroll Params
    Theta = roll_params(nn_weights, layers)
    
    # You need to return the following variables correctly 
    J = 0
    #Theta_grad = [zeros(w.shape) for w in Theta]
    
    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a 
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = zeros((num_labels, m))
    for i in range(m):
        yv[y[i],i] = 1.0

    # feedforward
    activation = transpose(concatenate((ones((m,1)), X), axis=1))
    activations = [activation]
    zs = [] # list to store all the z vectors, layer by layer
    for i in range(num_layers-1):
        z = dot(Theta[i], activation)
        zs.append(z)
        if i == (num_layers-2): #Final layer
            activation = sigmoid(z)
        else:
            activation = concatenate((ones((1,m)), sigmoid(z)), axis=0)

        activations.append(activation)
    
    # Cost Function
    J = (1.0/m)*(sum(-1.0*yv*log(activations[-1]) - (1.0 - yv)*log(1.0 - activations[-1])))
    for i  in range(num_layers-1):
        J = J + (l/(2.0*m))*sum(pow(Theta[i][:,1:],2.0))
    
    return J

    

