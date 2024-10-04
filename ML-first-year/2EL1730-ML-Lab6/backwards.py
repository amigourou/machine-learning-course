from numpy import *
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params

def backwards(nn_weights, layers, X, y, num_labels, lambd):

    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Unroll Params
    Theta = roll_params(nn_weights, layers)
  
    # You need to return the following variables correctly 
    Theta_grad = [zeros(w.shape) for w in Theta]
    
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
   
    #backward pass
    delta = activations[-1] - yv
    Theta_grad[-1] = (1.0 / m)*dot(delta, activations[-2].transpose())
    Theta_grad[-1][:,1:] = Theta_grad[-1][:,1:] + (lambd/m)*Theta[-1][:,1:]
    for l in range(2, num_layers):
        delta = dot(Theta[-l+1].transpose(), delta)[1:,:]*sigmoidGradient(zs[-l])
        Theta_grad[-l] = (1.0 / m)*dot(delta, activations[-l-1].transpose())
        Theta_grad[-l][:,1:] = Theta_grad[-l][:,1:] + (lambd/ m)*(Theta[-l][:,1:])

    Theta_grad = unroll_params(Theta_grad)
    
    return Theta_grad

    
