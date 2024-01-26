import numpy as np
from sigmoid import sigmoid

def computeGrad(theta, X, y):
    # Computes the gradient of the cost with respect to
    # the parameters.
    m = X.shape[0] # number of training examples
    grad = np.zeros(theta.shape) # initialize gradient

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of cost for each theta,
    # as described in the assignment.

    # grad = (1/m) * ((sigmoid(X.dot(theta)) - y).T.dot(X))
    for i in range(theta.shape[0]):
        for j in range(m):
            grad[i] += (sigmoid(np.dot(X[j, :],theta)) - y[j]) * X[j,i]
    grad /= m
    # =============================================================
    return grad.T
