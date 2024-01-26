import numpy as np
from sigmoid import sigmoid

def predict(theta, X):
    # Predict whether the label is 0 or 1 using learned logistic
    # regression parameters theta. The threshold is set at 0.5
    m = X.shape[0] # number of training examples
    c = np.zeros(m) # predicted classes of training examples
    p = np.zeros(m) # logistic regression outputs of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Predict the label of each instance of the
    #               training set.

    p = X.dot(theta)
    for i,_p in enumerate(p):
        if _p > 0.5 :
            c[i] += 1


    # =============================================================
    return c
