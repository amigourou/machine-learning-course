import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2
from generateData import generateData
from findClosestNeighbours import findClosestNeighbours
from spectralClustering import spectralClustering
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler

# Generate data
data = generateData()

# Plot data
plt.figure(1)
plt.scatter(data[:,0], data[:,1])  
plt.xlabel('x1')  
plt.ylabel('x2')  
plt.ylim(-6, 6)  
plt.xlim(-6, 6) 
plt.show()

# Number of clusters
k = 3

# Cluster using kmeans
centroids, labels = kmeans2(data, k)

# Plot clustering produced by kmeans
plt.figure(2)
plt.scatter(data[:,0], data[:,1], c=labels, facecolors="none") 
plt.xlabel('x1')  
plt.ylabel('x2') 
plt.title('Clustering by k-means algorithm')
plt.ylim(-6, 6)  
plt.xlim(-6, 6) 
plt.show()

# Find N closest neighbours of each data point
N = 10
closestNeighbours = findClosestNeighbours(data, N)

# Create adjacency matrix
W = np.zeros((data.shape[0], data.shape[0]))
for i in range(data.shape[0]):
    for j in range(N):
        W[i,int(closestNeighbours[i,j])] = 1
        W[int(closestNeighbours[i,j]),i] = 1
        

# Perform spectral clustering
labels = spectralClustering(W, k)  

# Plot clustering produced by spectral clustering
plt.figure(3)
plt.scatter(data[:,0], data[:,1], c=labels, facecolors="none") 
plt.xlabel('x1')  
plt.ylabel('x2')
plt.title('Clustering by Spectral Clustering algorithm')
plt.ylim(-6, 6)  
plt.xlim(-6, 6) 
plt.show()
