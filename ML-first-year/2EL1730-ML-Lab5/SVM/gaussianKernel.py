import numpy as np

def gaussianKernel(X1, X2, sigma = 0.1):
    m = X1.shape[0]
    K = np.zeros((m,X2.shape[0]))
    
    # ====================== YOUR CODE HERE =======================
    # Instructions: Calculate the Gaussian kernel (see the assignment
    #				for more details).
    
    for k in range(m) : 
        K[k,:] = np.exp(-np.linalg.norm(X1[k,:]-X2, axis = 1)**2/2*sigma**2)
    
    # =============================================================

    return K
