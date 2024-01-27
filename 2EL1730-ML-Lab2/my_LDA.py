import numpy as np
import scipy.linalg as linalg

def my_LDA(X, Y):
    """
    Train a LDA classifier from the training set
    X: training data
    Y: class labels of training data

    """    
    classLabels = np.unique(Y)  # different class labels on the dataset
    classNum = len(classLabels)
    datanum, dim = X.shape  # dimensions of the dataset
    totalMean = np.mean(X, 0)  # total mean of the data
    K = len(classLabels)
    Sw = np.zeros((dim,dim))
    Sb = np.zeros((dim,dim))
    centroid = np.zeros((K,dim))


    # ====================== YOUR CODE HERE ======================
    # Instructions: Implement the LDA technique, following the
    # steps given in the pseudocode on the assignment.
    # The function should return the projection matrix W,
    # the centroid vector for each class projected to the new
    # space defined by W and the projected data X_lda.

    for j in range(1,K+1) :
        Xj = X[np.where(Y == j),:].squeeze()
        n = len(Xj)
        mj = np.zeros((dim,1))
        
        for _,x in enumerate(Xj) :
            mj +=np.expand_dims(x,axis = 1)
        mj/=n
        centroid[j-1,:] += mj.squeeze()
        for x in Xj :
            Sw += (x-mj).dot((x-mj).T)
        
        Sb+= n * (mj-totalMean).dot((mj-totalMean).T)
    
    lambdas, W = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    X_lda = np.dot(X,W)
    projected_centroid = np.dot(centroid,W)

    # =============================================================

    return W, projected_centroid, X_lda
