import numpy as np

def findClosestNeighbours(data, N):
    
    closestNeighbours = np.zeros((data.shape[0], N))
    distances = np.zeros(data.shape[0])

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i != j:
                distances[j] = np.linalg.norm(data[i,:] - data[j,:])    
            else:
                distances[j] = 0
                
        closestNeighbours[i,:] = np.argsort(distances)[:N]
    
    return closestNeighbours
