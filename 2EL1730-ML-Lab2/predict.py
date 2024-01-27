import numpy as np

def predict(X, projected_centroid, W):
    """Apply the trained LDA classifier on the test data
    X: test data
    projected_centroid: centroid vectors of each class projected to the new space
    W: projection matrix computed by LDA
    """

    # Project test data onto the LDA space defined by W
    projected_data = np.dot(X, W)
    label = []

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in the code to implement the classification
    # part of LDA. Follow the steps given in the assigment.

    for y in projected_data :
        l = 0
        dists = []
        for i in range(projected_centroid.shape[0]):
            dists.append(np.linalg.norm(y-projected_centroid[i,:]))
        l = dists.index(np.min(dists))
        label.append(l)


    # =============================================================

    # Return the predicted labels of the test data X
    return np.array(label)
