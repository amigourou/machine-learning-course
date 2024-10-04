import numpy as np

def kNN(k, X, labels, y):
    # Assigns to the test instance the label of the majority of the labels of the k closest 
	# training examples using the kNN with euclidean distance.
    #
    # Input: k: number of nearest neighbors
    #        X: training data           
    #        labels: class labels of training data
    #        y: test data
    
    
    # ====================== ADD YOUR CODE HERE =============================
    # Instructions: Run the kNN algorithm to predict the class of
    #               y. Rows of X correspond to observations, columns
    #               to features. The 'labels' vector contains the 
    #               class to which each observation of the training 
    #               data X belongs. Calculate the distance betweet y and each 
    #               row of X, find  the k closest observations and give y 
    #               the class of the majority of them.
    #
    # Note: To compute the distance betweet two vectors A and B use
    #       use the np.linalg.norm() function.
    #

    distances = np.zeros((X.shape[0],1))
    for j,x in enumerate(X) :
        distances[j] += np.linalg.norm(x-y)

    distances = distances.squeeze()
    indices_sorted = np.argsort(distances)
    distances.sort()

    l = np.zeros(len(np.unique(labels)))
    for i in range(k):
        l[labels[indices_sorted[i]]] += 1
    
    label = list(l).index(np.max(l))

    # return the label of the test data
    
    return label
