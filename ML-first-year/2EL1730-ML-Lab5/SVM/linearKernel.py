import numpy as np

def linearKernel(X1, X2):
    # Computes the linear Kernel between two set of features
    m = X1.shape[0]
    K = np.zeros((m,X2.shape[0]))
    
    # ====================== YOUR CODE HERE =======================
    # Instructions: Calculate the linear kernel (see the assignment
    #				for more details).
    
    for k in range(m):
        K[k,:]  = np.dot(X2, X1[k,:])
    
    # =============================================================
        
    return K
