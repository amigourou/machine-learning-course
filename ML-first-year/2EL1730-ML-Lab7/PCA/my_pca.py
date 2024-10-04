import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab
from numpy import genfromtxt
import csv

# Load the data set (wine). Variable data stores the final data (178 x 13)
my_data = np.genfromtxt(r'D:\Deep learning projects\machine-learning-course\2EL1730-ML-Lab7\PCA\wine_data.csv', delimiter=',')
data = my_data[:,1:]
target= my_data[:,0] # Class of each instance (1, 2 or 3)
print("Size of the data {} ".format(data.shape))

# Draw the data in 3/13 dimensions (Hint: experiment with different combinations of dimensions)
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(data[:,3],data[:,1],data[:,2], c=target)
ax1.set_xlabel('1st dimension')
ax1.set_ylabel('2nd dimension')
ax1.set_zlabel('3rd dimension')
ax1.set_title("Vizualization of the dataset (3 out of 13 dimensions)")


A = data
#================= ADD YOUR CODE HERE ====================================
# Performs principal components analysis (PCA) on the n-by-p data matrix A (data)
# Rows of A correspond to observations (i.e., wines), columns to variables.
## TODO: Implement PCA
# Instructions: Perform PCA on the data matrix A to reduce its
#				dimensionality to 2 and 3. Save the projected
#				data in variables newData2 and newData3 respectively
C = np.zeros(A.shape)
for i in range(A.shape[1]):
    C[:,i] = A[:,i]-np.ones(A[:,i].shape) * np.mean(A[:,1])

W = np.dot(C.T,C)
values, vectors = np.linalg.eigh(W)

idx = np.argsort(values)[::-1]
values, vectors = values[idx], vectors[:,idx]

newData2 = np.dot(C,vectors[:,:2])
newData3 = np.dot(C,vectors[:,:3])




#=============================================================================

# # Plot the first two principal components 
# plt.figure(2)
# plt.scatter(newData2[:,0],newData2[:,1], c=target)
# plt.xlabel('1st Principal Component')
# plt.ylabel('2nd Principal Component')
# plt.title("Projection to the top-2 Principal Components")
# plt.draw()

# Plot the first three principal components 
fig = plt.figure(3)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(newData3[:,0],newData3[:,1], newData3[:,2], c=target)
ax.set_xlabel('1st Principal Component')
ax.set_ylabel('2nd Principal Component')
ax.set_zlabel('3rd Principal Component')
ax.set_title("Projection to the top-3 Principal Components")
plt.show()  
