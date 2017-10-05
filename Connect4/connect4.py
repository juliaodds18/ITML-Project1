############################# Imports #############################
import scipy as sp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.cross_validation import cross_val_score
import sklearn.preprocessing as pp
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import json
import os
from sklearn.tree import DecisionTreeClassifier as DTC

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
print(countInstances)

# Print one attrbute and instances of that attribute
connect4_role0.head()

# Average, standard dev, median of values
print(connect4_role0.describe())
connect4_role0.head()

############################# Preprocessing the dataset #############################

# Convert values to numberi
le = pp.LabelEncoder()
for col in connect4_role0.columns:
    connect4_role0[col] = le.fit_transform(connect4_role0[col])
    
# Print the head of the dataset to check if it is numeric
connect4_role0.head()

############################# Splitting the data to traingin and test set #############################

data = connect4_role0.values

Y = data[:,0]
X = data[:,1:23]

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=42)

#sns.countplot(Y)
#plt.show()