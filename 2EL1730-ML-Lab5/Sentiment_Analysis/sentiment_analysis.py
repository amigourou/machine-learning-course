
# Import python modules
from cgi import test
import numpy as np
import csv
import matplotlib.pyplot as plt
import nltk
from nltk.stem.porter import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection  import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

#%% 
# Part 1: Load data from .csv file
############
with open(r'D:\Deep learning projects\machine-learning-course\2EL1730-ML-Lab5\Sentiment_Analysis\movie_reviews.csv', encoding= 'utf-8') as csv_file:
    reader = csv.reader(csv_file, delimiter=',',quotechar='"')
        
    # Initialize lists for data and class labels
    data =[]
    labels = []
    # For each row of the csv file
    for row in reader:
        # skip missing data
        if row[0] and row[-1]:
            data.append(row[0])
            y_label = -1 if row[-1]=='negative' else 1
            labels.append(y_label)


#%%
# Part 2: Data preprocessing
############
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

# For each document in the dataset, do the preprocessing
for doc_id, text in enumerate(data):

    punctuation = set(string.punctuation)
    doc = [w for w in text.lower() if w not in punctuation]

    doc = [w for w in doc if w not in stopwords]

    stemmer = PorterStemmer()
    doc = [stemmer.stem(w) for w in doc]

    # Convert list of words to one string
    doc = ''.join(w for w in doc)
    data[doc_id] = doc   # list data contains the preprocessed documents



#%%
# Part 3: Feature extraction and the TF-IDF matrix
m = TfidfVectorizer()
tfidf_matrix = m.fit_transform(data)
tfidf_matrix = tfidf_matrix.toarray()

#%%
# Part 4: Model learning and prediction
#############
data_train, data_test, labels_train, labels_test = train_test_split(
    tfidf_matrix, labels, test_size = 0.4, random_state = 42)



## Model learning and prediction
clf = svm.LinearSVC()
y_score = clf.fit(data_train,labels_train)
       
#### Evaluation of the prediction

labels_predicted = clf.predict(data_test)
print(classification_report(labels_test, labels_predicted))
print("Accuracy: {:.2%}".format(accuracy_score(labels_test, labels_predicted)))
