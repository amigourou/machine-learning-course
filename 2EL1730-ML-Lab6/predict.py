from numpy import *
from sigmoid import sigmoid

def predict(Theta, X):

    # Useful values
    m = X.shape[0]
    num_labels = Theta[-1].shape[0]
    num_layers = len(Theta) + 1
        
    # You need to return the following variables correctly
    p = zeros((1,m))
    h = X
    activation = transpose(concatenate((ones((m,1)), X), axis=1))
    for i in range(num_layers - 1):       
        z =  dot(Theta[i], activation)
        if i == (num_layers-2):
            activation = sigmoid(z)
        else:
            activation = concatenate((ones((1,m)), sigmoid(z)), axis=0)
    p = argmax(activation,axis = 0)
    
    return p

