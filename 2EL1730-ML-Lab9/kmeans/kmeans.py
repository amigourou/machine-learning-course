import numpy as np
from simpleInitialization import simpleInitialization
import matplotlib.pyplot as plt

def kmeans(X, k):
    # Intialize centroids
    centroids = simpleInitialization(X, k)
    labels = np.zeros(X.shape[0], dtype = np.int32)
     # ====================== ADD YOUR CODE HERE ======================
    # Instructions: Run the main k-means algorithm. Follow the steps 
    #               given in the description. Compute the distance 
    #               between each instance and each centroid. Assign 
    #               the instance to the cluster described by the closest
    #               centroid. Repeat the above steps until the centroids
    #               stop moving or reached a certain number of iterations
    #               (e.g., 100).

    initialCentroidsIndices = np.random.choice(range(X.shape[0]),k, replace = False)
    initialCentroids = X[initialCentroidsIndices]
    centroids = initialCentroids

    for i in range(100) :
        for m in range(X.shape[0]) :
            distances = [np.linalg.norm(X[m] - centroids[d]) for d in range(k)]
            cm = np.where(distances == min(distances))[0]
            cm = int(cm.item())
            labels[m] = cm
        
        for n in range(k) :
            samplesClusterN = list(np.where(labels == n)[0])
            centroids[n] =(1/len(samplesClusterN)) * np.sum(X[samplesClusterN],axis = 0)

    
    # ===============================================================
    return labels

if __name__ == '__main__' :
    k = 4
    X = np.random.rand(100,2)
    labels = kmeans(X,k)

    for i in range(k) :
        plt.scatter(X[np.where(labels == i),0],X[np.where(labels == i),1], label = f'{i}')
    plt.show()