from numpy import *
from read_dataset import read_dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from displayData import displayData
from randInitializeWeights import randInitializeWeights
from costFunction import costFunction
from unroll_params import unroll_params
from roll_params import roll_params
from scipy.optimize import *
from predict import predict
from backwards import backwards
from checkNNCost import checkNNCost
from checkNNGradients import checkNNGradients
from fmincgb2 import fmincgb2
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient



#=========== Step 1: Loading and Visualizing Data =============
print("\nLoading and visualizing Data ...\n")

#Reading of the dataset
size_training = 50    # number of samples retained for training
size_test     = 100     # number of samples retained for testing
images_training, labels_training, images_test, labels_test = read_dataset(size_training, size_test)

# Randomly select 100 data points to display
random_instances = list(range(size_training))
random.shuffle(random_instances)
displayData(images_training[random_instances[0:100],:])

input('Program paused. Press enter to continue!!!')

#================ Step 2: Setting up Neural Network Structure  ================
print("\nSetting up Neural Network Structure ...\n")

# Setup the parameters you will use for this exercise
input_layer_size   = 784        # 28x28 Input Images of Digits
num_labels         = 10         # 10 labels, from 1 to 10 (note that we have mapped "0" to label 10)

num_of_hidden_layers = int(input('Please select the number of hidden layers: '))
print("\n")

layers = [input_layer_size]
for i in range(num_of_hidden_layers):
    layers = layers +  [int(input('Please select the number nodes for the ' + str(i+1) + ' hidden layers: '))]
layers = layers + [num_labels]

input('Program paused. Press enter to continue!!!')

# ================ Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print("\nInitializing Neural Network Parameters ...\n")

Theta = randInitializeWeights(layers)
# Unroll parameters
nn_weights = unroll_params(Theta)

input('Program paused. Press enter to continue!!!')

# ================ Sigmoid  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print("\nEvaluating sigmoid function ...\n")

g = sigmoid(array([1, -0.5, 0,  0.5, 1]))
print("Sigmoid evaluated at [1 -0.5 0 0.5 1]:  ")
print(g)

input('Program paused. Press enter to continue!!!')

# ================ Sigmoid Gradient ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print("\nEvaluating Sigmoid Gradient function ...\n")

g = sigmoidGradient(array([1, -0.5, 0,  0.5, 1]))
print("Sigmoid evaluated at [1 -0.5 0 0.5 1]:  ")
print(g)

input('Program paused. Press enter to continue!!!')

# ================ Implement Feedforward (Cost Function) ================

print("\n Checking Cost Function without Regularization (Feedforward) ...\n")

lambd = 0.0
checkNNCost(lambd)

input('Program paused. Press enter to continue!!!')

# ================ Implement Feedforward with Regularization  ================

print("\n Checking Cost Function with Reguralization ... \n")

lambd = 3.0
checkNNCost(lambd)

input('Program paused. Press enter to continue!!!')

# ================ Implement Backpropagation  ================

print("\n Checking Backpropagation without Regularization ...\n")

lambd = 0.0
checkNNGradients(lambd)
input('Program paused. Press enter to continue!!!')


# ================ Implement Backpropagation with Regularization  ================

print("\n Checking Backpropagation with Regularization ...\n")

lambd = 3.0
checkNNGradients(lambd)

input('Program paused. Press enter to continue!!!')


#================ Training Neural Networks  ================
print("\nTraining Neural Network... \n")

#  You should also try different values of the regularization factor
lambd = 3.0

res = fmin_l_bfgs_b(costFunction, nn_weights, fprime = backwards, args = (layers,  images_training, labels_training, num_labels, 1.0), maxfun = 1000, factr = 1., disp = True)
Theta = roll_params(res[0], layers)

input('Program paused. Press enter to continue!!!')

# ================ Implement Prediction - Testing  ================
print("\nTesting Neural Network... \n")

pred  = predict(Theta, images_test)
print('\n Accuracy: ' + str(mean(labels_test==pred) * 100))

