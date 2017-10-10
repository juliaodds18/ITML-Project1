############################# Imports #############################
import scipy as sp
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import cross_val_score
import sklearn.preprocessing as pp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import json
import os
import datetime
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
import sklearn.pipeline 
from sklearn.grid_search import GridSearchCV
from sklearn import grid_search

############################# Data exploration #############################
 
# Import the dataset
index=[]
board = ['a','b','c','d','e','f','g']
for i in board:
    for j in range(6):
        index.append(i + str(j+1))
 
column_names  = index +['Class']
connect4_role0 = pd.read_csv("connect-4.data")
 
# Print instances
countInstances = (connect4_role0.count())
#print(countInstances)
 
# Print one attrbute and instances of that attribute
connect4_role0.head()
 
# Average, standard dev, median of values
#print(connect4_role0.describe())
connect4_role0.head()
 
############################# Preprocessing the dataset #############################
 
# Convert values to numberi
le = pp.LabelEncoder()
for col in connect4_role0.columns:
   connect4_role0[col] = le.fit_transform(connect4_role0[col])
 
for col in connect4_role0.columns:
    connect4_role0[col] = pd.get_dummies(connect4_role0[col])
 
# Print the head of the dataset to check if it is numeric
connect4_role0.head()
 
############################# Splitting the data to traingin and test set #############################
 
# Get the values in the dataset
data = connect4_role0.values
 
# Split the dataset in input and output
Y = data[:,0]
X = data[:,1:23]
 
# Split the data into training an test sets
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=42)
 
# Visualize the output
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
sns.countplot(Y)
plt.ioff()
plt.draw()
plt.savefig('find_best_k', dpi=300)
plt.close()

############################# Decision Tree Classifier #############################

# Make a list for possible max depth, min sample split and 
# min sample leaf
listOfparameters = list(range(2,4000))
k = list(filter(lambda x: x % 2 == 0, listOfparameters))

# Initialize the Decision tree classifier
clf = DTC()
steps = [
    ('clf', clf)
]

# Initialize the parameters for the Decision tree classifier
parameters = {
    'clf__max_depth': (listOfparameters),
    'clf__min_samples_split': (listOfparameters),
    'clf__min_samples_leaf': (listOfparameters)
}

# Use pipline an gridSearchCv to preidict the best 
# values for the parameters 
pipeline = sklearn.pipeline.Pipeline(steps)
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
y_predictions = cv.predict(X_test)
report = classification_report(y_test, y_predictions)

# Print out the best value for the Decision tree classifier
print(report)
print(cv.best_params_)


# select features
k=10
feature_selector = SelectKBest(k=k)
X_train = feature_selector.fit_transform(X_train, y_train)
X_test = feature_selector.transform(X_test)
#feature_names = [ feature_names[i] for i in feature_selector.get_support(indices=True) ]
print("features selected: %d" % X_train.shape[1])

# train decision tree
dt = DTC(min_samples_split=2)
dt.fit(X_train, y_train)

# report accuracy
print("the decision tree has %d nodes" % dt.tree_.node_count)
print("train accuracy: %f" % dt.score(X_train, y_train))
print("test accuracy: %f" % dt.score(X_test, y_test))


############################# k nearest neighbor #############################

# Best value for k
n_neighbors = 29

# List of k to find the best 
neigh = list(range(1,4000))

#Find the current directory 
directory = os.getcwd()

# Open a new grading file (truncates if exists, change that by erasing the + in 'w+')
textFile = open('bestK_3.txt', 'w+')

# Find k with best accuracy 
for k in neigh:
	knn = KNeighborsClassifier(n_neighbors=k)
	scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')

        # Read the value into a json string 
	jsonObj = {'score:' : scores.mean(), 'k:' : k}
	textFile.write(json.dumps(jsonObj, ensure_ascii = False) + "\n")



textFile.close()

# knn = KNeighborsClassifier(n_neighbors=3)
knn = KNeighborsClassifier(n_neighbors=29)
clf_knn=knn.fit(X_train, y_train)
print ("Acurracy: ", clf_knn.score(X_test,y_test))


#Find the current directory 
# directory = os.getcwd()
# Open a new grading file (truncates if exists, change that by erasing the + in 'w+')
#jsonFile = open('bestK_2.txt', 'r')

data = []
with open('bestK_2.txt','r') as f:
    for line in f:
       data.append(json.loads(line))

for line in data: 
    accuracy.append(line['score:'])
    ks.append(line['k:'])

# Plot the k and the accuracy
plt.title("Find the best k")
plt.xlabel("k")
plt.ylabel("Accuracy")
    
plt.plot(ks, accuracy)
plt.ioff()
plt.draw()
plt.savefig('find_best_k', dpi=300)
plt.close()

############################# SVM #############################

# Initialize the SVM
clf = SVC()

# List of gamma and C to find the best
gamma = np.arange(0.001, 1.0, 0.001)
listOfparameters = list(range(1,500))
k = list(filter(lambda x: x % 2 == 0, listOfparameters))
steps = [
    ('clf', clf)
]
 
#Initialize the parameters for the SVM
parameters = {
    'clf__C': (listOfparameters),
    'clf__gamma': (0.1, 0.001, 0.0001),
}

# Use pipline an gridSearchCv to preidict the best 
# values for the parameters  
pipeline = Pipeline(steps) 
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
y_predictions = cv.predict(X_test)
report = classification_report(y_test, y_predictions)
 
# Print the information for SVM
print(report)
print(cv.best_params_)

# Train the Support vector classifier and print it
svc = SVC(kernel='rbf', gamma=0.1, C=)
clf_svc=svc.fit(X_train, y_train)
print ("Acurracy: ", clf_svc.score(X_test,y_test) )


