import numpy as np

def sigmoid(z):
    # Computes the sigmoid of z.
    # ====================== YOUR CODE HERE ======================
    # Instructions: Implement the sigmoid function as given in the
    # assignment.

    g = 1/(1+np.exp(-z))

    # =============================================================

    return g
