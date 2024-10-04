
# coding: utf-8

# https://www.python.org/dev/peps/pep-0008#introduction<BR>
# http://scikit-learn.org/<BR>
# http://pandas.pydata.org/<BR>

#%%
import numpy as np
import pandas as pd
import pylab as plt
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


### Fetch the data and load it in pandas
data = pd.read_csv(r'D:\Deep learning projects\machine-learning-course\2EL1730-ML-Lab4\train.csv')
print("Size of the data: ", data.shape)

#%%
# See data (five rows) using pandas tools
#print data.head(2)


### Prepare input to scikit and train and test cut

binary_data = data[np.logical_or(data['Cover_Type'] == 1, data['Cover_Type'] == 2)] # two-class classification set
X = binary_data.drop('Cover_Type', axis=1).values
y = binary_data['Cover_Type'].values
print(np.unique(y))
y = 2 * y - 3 # converting labels from [1,2] to [-1,1]

#%%
# Import cross validation tools from scikit
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)


#%%
### Train a single decision tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=8)

# Train the classifier and print training time
clf.fit(X_train, y_train)

#%%
# Do classification on the test dataset and print classification results
from sklearn.metrics import classification_report
target_names = data['Cover_Type'].unique().astype(str).sort()
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target_names))

#%%
# Compute accuracy of the classifier (correctly classified instances)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


#===================================================================
#%%
### Train AdaBoost

# Your first exercise is to program AdaBoost.
# You can call *DecisionTreeClassifier* as above, 
# but you have to figure out how to pass the weight vector (for weighted classification) 
# to the *fit* function using the help pages of scikit-learn. At the end of 
# the loop, compute the training and test errors so the last section of the code can 
# plot the lerning curves. 
# 
# Once the code is finished, play around with the hyperparameters (D and T), 
# and try to understand what is happening.

def train_AdaBoost(T,D) : 
    w = np.ones(X_train.shape[0]) / X_train.shape[0] # weight initialization
    training_scores = np.zeros(X_train.shape[0]) # init scores with 0
    test_scores     = np.zeros(X_test.shape[0])

    # init errors
    training_errors = []
    test_errors = []

    #===============================
    for t in range(T):
        clf = DecisionTreeClassifier(max_depth=D)
        try : 
            yt = clf.fit(X_train, y_train, sample_weight=w)
            train_preds = yt.predict(X_train)
            test_preds = yt.predict(X_test)
            gammat = 0
            indicator = np.not_equal(train_preds, y_train)
            gammat = np.sum(w[indicator])/np.sum(w)
            alphat = (1/2)*np.log((1-gammat)/gammat)

            w *= np.exp(alphat * indicator)
            # calculate scores and errors for this tree
            training_scores += alphat * train_preds
            training_error = 1. * len(training_scores[training_scores * y_train < 0]) / len(X_train)
            # calculate test error and score
            test_scores += alphat * test_preds
            test_error = 1. * len(test_scores[test_scores * y_test < 0]) / len(X_test)
            #print(t, ": ", alpha, gamma, training_error, test_error)

            training_errors.append(training_error)
            test_errors.append(test_error)
        except Exception :
            pass

    return training_errors, test_errors

#===============================

#  Plot training and test error    

# plt.legend()

#===================================================================
#%%
### Optional part
### Optimize AdaBoost

# Your final exercise is to optimize the tree depth in AdaBoost. 
# Copy-paste your AdaBoost code into a function, and call it with different tree depths 
# and, for simplicity, with T = 100 iterations (number of trees). Plot the final 
# test error vs the tree depth. Discuss the plot.

#===============================

# Your code should go here
final_scores_train = []
final_scores_test = []
for  D in tqdm(range(1,15)) :
    training_errors, test_errors = train_AdaBoost(100,D)
    final_scores_train.append(training_errors[-1])
    final_scores_test.append(test_errors[-1])
plt.plot(final_scores_train, label = 'training')
plt.plot(final_scores_test, label = 'test')
plt.show()

#===============================
