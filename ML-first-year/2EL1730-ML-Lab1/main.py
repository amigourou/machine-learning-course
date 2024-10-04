import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from computeCost import computeCost
from computeGrad import computeGrad
from predict import predict

# Load the dataset
# The first two columns contains the exam scores and the third column
# contains the label.
data = np.loadtxt(r'C:\Users\Emilien\DL projects\machine-learning-course\2EL1730-ML-Lab1\data1.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]

# Plot data
pos = np.where(y == 1)
neg = np.where(y == 0)
plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not Admitted'])
# plt.show()


#Add intercept term to X
X_new = np.ones((X.shape[0], 3))
X_new[:, 1:3] = X
X = X_new

# y = np.expand_dims(y,axis = 1)
###-----------Test of implementations---------
def test_computeCost():
    # test 1
    theta, cost = [0, 0, 0], 0.6931471
    if np.abs(computeCost(np.expand_dims(np.array(theta),axis = 1),X,y)-cost) < 1e-5:
        print("test 1 passed")
    else:
        raise ValueError("test 1 not passed!")

    # test 2
    theta, cost = [-0.01, 0.05, 0], 1.092916
    if np.abs(computeCost(np.expand_dims(np.array(theta),axis = 1),X,y)-cost) < 1e-5:
        print("test 2 passed")
    else:
        raise ValueError("test 2 not passed!")

test_computeCost()

def test_computeGrad():
    # test 1
    theta, grad = np.expand_dims(np.array([0, 0, 0]),axis = 1), np.array([ -0.1, -12.00921659, -11.26284221])
    if np.sum(np.abs(computeGrad(theta, X ,y)-grad)) < 1e-5:
        print("test 1 passed")
    else:
        raise ValueError("test 1 not passed!")

    # test 2
    theta, grad = np.expand_dims(np.array([0.02, 0, -0.04]),axis = 1), np.array([-0.51775522, -39.39901278, -39.85199474])
    if np.sum(np.abs(computeGrad(theta, X, y)-grad)) < 1e-5:
        print("test 2 passed")
    else:
        raise ValueError("test 2 not passed!")

test_computeGrad()

###----End of tests----

# Initialize fitting parameters

initial_theta = np.zeros((3,1)).squeeze()

# Run minimize() to obtain the optimal theta
Result = op.minimize(fun = computeCost, x0 = initial_theta, args = (X, y), method = 'TNC',jac = computeGrad)
theta = Result.x

# Plot the decision boundary
plot_x = np.array([min(X[:, 1]), max(X[:, 1])])
plot_y = (- 1.0 / theta[2]) * (theta[1] * plot_x + theta[0])
plt.plot(plot_x, plot_y)
plt.scatter(X[pos, 1], X[pos, 2], marker='o', c='b')
plt.scatter(X[neg, 1], X[neg, 2], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Decision Boundary', 'Admitted', 'Not Admitted'])
plt.show()

# Compute accuracy on the training set
p = predict(np.array(theta), X)
counter = 0
for i in range(y.size):
    if p[i] == y[i]:
        counter += 1
print('Train Accuracy: {:.2f}'.format(counter / float(y.size) * 100.0))
