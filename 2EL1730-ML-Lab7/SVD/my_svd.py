import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

# Load the "gatlin" image data
X = np.loadtxt(r'D:\Deep learning projects\machine-learning-course\2EL1730-ML-Lab7\SVD\gatlin.csv', delimiter=',')
print(X)

#=========================================================================
# Perform SVD decomposition
# xxt = np.dot(X,X.T)
# xtx = np.dot(X.T,X)

# ev,U = np.linalg.eigh(xxt)

# sigma = np.eye(N = X.shape[0], M = X.shape[1])
# for i  in range(len(ev)):
#     sigma[i,i]*=np.sqrt(ev[i])

# ev2,V = np.linalg.eigh(xtx)

U,sigma,V = np.linalg.svd(X)



#=========================================================================

# Plot the original image
plt.figure(1)
plt.imshow(X, cmap=cm.Greys_r)
plt.title('Original image (rank 480)')
plt.axis('off')
plt.show()


#=========================================================================
# Matrix reconstruction using the top k = [10, 20, 50, 100, 200] singular values

X_ks = []
for k in [10, 20, 50, 100, 200, 300, 480] :
    X_ks.append(np.dot(U[:,:k],np.dot(np.diag(sigma)[:k,:k],V[:k,:])))
    plt.plot(sigma, label = f'{k}')
plt.show()
    

#=========================================================================

#=========================================================================
# Error of approximation
errors = []
for X_k in X_ks:
    errors.append(np.linalg.norm(X_k-X)/np.linalg.norm(X))


#=========================================================================

# Plot the optimal rank-k approximation for various values of k)
# Create a figure with 6 subfigures
plt.figure(2)   

# Rank 10 approximation
plt.subplot(321)
plt.imshow(X_ks[0], cmap=cm.Greys_r)
plt.title('Best rank' + str(10) + ' approximation')
plt.axis('off')

# Rank 20 approximation
plt.subplot(322)
plt.imshow(X_ks[1], cmap=cm.Greys_r)
plt.title('Best rank' + str(20) + ' approximation')
plt.axis('off')

# Rank 50 approximation
plt.subplot(323)
plt.imshow(X_ks[2], cmap=cm.Greys_r)
plt.title('Best rank' + str(50) + ' approximation')
plt.axis('off')

# Rank 100 approximation
plt.subplot(324)
plt.imshow(X_ks[3], cmap=cm.Greys_r)
plt.title('Best rank' + str(100) + ' approximation')
plt.axis('off')

# Rank 200 approximation
plt.subplot(325)
plt.imshow(X_ks[5], cmap=cm.Greys_r)
plt.title('Best rank' + str(200) + ' approximation')
plt.axis('off')



# Original
plt.subplot(326)
plt.imshow(X, cmap=cm.Greys_r)
plt.title('Original image (Rank 480)')
plt.axis('off')



plt.show()
