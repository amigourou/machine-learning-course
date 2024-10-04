import numpy as np
from sigmoid import sigmoid

def computeCost(theta : np.array, X : np.array, y : np.array):
    # Computes the cost using theta as the parameter
    # for logistic regression.
    m = X.shape[0] # number of training examples
    # ====================== YOUR CODE HERE ======================
    # Instructions: Calculate the error J of the decision boundary
    #               that is described by theta (see the assignment
    #               for more details).
    J = -(1/m)*(y.T.dot(np.log(sigmoid(X.dot(theta)))) + (1-y).T.dot(np.log(1-sigmoid(X.dot(theta)))))

    # =============================================================
    return J
