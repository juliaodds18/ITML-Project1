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

############################# Data exploration #############################

# Import the dataset
mushrooms = pd.read_csv("mushrooms.csv")

# Print instances 
countInstances = (mushrooms.count())
print(countInstances)

# Print one attrbute and instances of that attribute
mushrooms.head()

# Average, standard dev, median of values
print(mushrooms.describe())
mushrooms.head()

mushrooms.describe()

############################# Preprocessing the dataset #############################

# Look if there is any missing value
for feature in mushrooms.columns:
    print(feature, ':', mushrooms[feature].unique())
    
# Delete stalk-root where the value is questionmark and the veil-type only contains one value 
mushrooms = mushrooms.drop(mushrooms[mushrooms['stalk-root']=='?'].index)
mushrooms = mushrooms.drop('veil-type', axis=1)


# Change the values to numeric with labelancoder
mushrooms_data = mushrooms.values
lEncoder = LabelEncoder()
lEncoder.fit(mushrooms_data[:, 0])
dataa = lEncoder.transform(mushrooms_data[:, 0])

# Loop through all the columns and change them to numeric
# Put all the new values into vstack
for ix in range(1, mushrooms_data.shape[1]):
    le = pp.LabelEncoder()
    le.fit(mushrooms_data[:, ix])
    y = le.transform(mushrooms_data[:, ix])
    dataa = np.vstack((dataa , y))
   
# Transform the  
data = dataa.T

print(data)

# Look if the missing value is gone 
for feature in mushrooms.columns:
    print(feature, ':', mushrooms[feature].unique())


############################# Splitting the data to traingin and test set #############################

Y = data[:,0]
X = data[:,1:23]

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=42)

sns.countplot(Y)
plt.show()

############################# K nearest neighbor #############################

'''listOfk = list(range(1,4000))
neigh = list(filter(lambda x: x % 10 == 0, listOfk))


#Find the current directory 
directory = os.getcwd()
# Open a new grading file (truncates if exists, change that by erasing the + in 'w+')
textFile = open('bestK.txt', 'w+')

accuracy = []
answers = []

for i in range(len(neigh)):
    knn = KNN(n_neighbors=neigh[i], n_jobs=-1)
    
    knn.fit(X_train, y_train)

    #print "Training Time : ", end-start
    score = knn.score(X_test, y_test)

    #print "Testing Time : ", end-start

    #print "Accurcy : ", score*100 
    jsonObj = {'score:' : score*100}
    textFile.write(json.dumps(jsonObj, ensure_ascii = False) + "\n")
    # accuracy.append(score*100)
    # temp = dt.feature_importances_
    # answers.append(temp)
    #print "\n" 

textFile.close()

temp = []

for i in range(0, len(ans)):
    temp.append(np.argmax(ans[ix]))

mode = max(set(temp), key=temp.count) #find mode for features importance in variable estimators
print "Features most indicative of a poisonous mushroom wrt kNN : ", headers[mode+1]

plt.figure(2)
plt.suptitle('k-Nearest Neighbour Plot', fontsize=10)
plt.plot(neighbours, acc, '-o')
plt.xlabel('Neighbours', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.show()'''

# Create color maps
'''cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
listOfk = list(range(1,4000))
neigh = list(filter(lambda x: x % 10 == 0, listOfk))

# empty list that will hold cv scores
# cv_scores = []

#Find the current directory 
directory = os.getcwd()
# Open a new grading file (truncates if exists, change that by erasing the + in 'w+')
textFile = open('bestK.txt', 'w+')

# Perform 10-fold cross validation
# This is done to find the best value for k 
for k in neigh:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    # cv_scores.append(scores.mean())
    jsonObj = {'k' : k, 'score:' : scores.mean()}
    textFile.write(json.dumps(jsonObj, ensure_ascii = False) + "\n")

# This is the best value for k, at least on the range that we tested
print(max(cv_scores))
textFile.close()'''

'''
Logical rules for the mushroom data sets.

	Logical rules given below seem to be the simplest possible for the
	mushroom dataset and therefore should be treated as benchmark results.

	Disjunctive rules for poisonous mushrooms, from most general
	to most specific:

	P_1) odor=NOT(almond.OR.anise.OR.none)
	     120 poisonous cases missed, 98.52% accuracy

	P_2) spore-print-color=green
	     48 cases missed, 99.41% accuracy
         
	P_3) odor=none.AND.stalk-surface-below-ring=scaly.AND.
	          (stalk-color-above-ring=NOT.brown) 
	     8 cases missed, 99.90% accuracy
         
	P_4) habitat=leaves.AND.cap-color=white
	         100% accuracy     

	Rule P_4) may also be

	P_4') population=clustered.AND.cap_color=white

	These rule involve 6 attributes (out of 22). Rules for edible
	mushrooms are obtained as negation of the rules given above, for
	example the rule:

	odor=(almond.OR.anise.OR.none).AND.spore-print-color=NOT.green

	gives 48 errors, or 99.41% accuracy on the whole dataset.

	Several slightly more complex variations on these rules exist,
	involving other attributes, such as gill_size, gill_spacing,
	stalk_surface_above_ring, but the rules given above are the simplest
	we have found.

	 1. cap-shape:                bell=b,conical=c,convex=x,flat=f,
                                  knobbed=k,sunken=s
     2. cap-surface:              fibrous=f,grooves=g,scaly=y,smooth=s
     3. cap-color:                brown=n,buff=b,cinnamon=c,gray=g,green=r,
                                  pink=p,purple=u,red=e,white=w,yellow=y
     4. bruises?:                 bruises=t,no=f
     5. odor:                     almond=a,anise=l,creosote=c,fishy=y,foul=f,
                                  musty=m,none=n,pungent=p,spicy=s
     6. gill-attachment:          attached=a,descending=d,free=f,notched=n
     7. gill-spacing:             close=c,crowded=w,distant=d
     8. gill-size:                broad=b,narrow=n
     9. gill-color:               black=k,brown=n,buff=b,chocolate=h,gray=g,
                                  green=r,orange=o,pink=p,purple=u,red=e,
                                  white=w,yellow=y
    10. stalk-shape:              enlarging=e,tapering=t
    11. stalk-root:               bulbous=b,club=c,cup=u,equal=e,
                                  rhizomorphs=z,rooted=r,missing=?
    12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
    13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
    14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                  pink=p,red=e,white=w,yellow=y
    15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o,
                                  pink=p,red=e,white=w,yellow=y
    16. veil-type:                partial=p,universal=u
    17. veil-color:               brown=n,orange=o,white=w,yellow=y
    18. ring-number:              none=n,one=o,two=t
    19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l,
                                  none=n,pendant=p,sheathing=s,zone=z
    20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r,
                                  orange=o,purple=u,white=w,yellow=y
    21. population:               abundant=a,clustered=c,numerous=n,
                                  scattered=s,several=v,solitary=y
    22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p,
                                  urban=u,waste=w,woods=d

'''

