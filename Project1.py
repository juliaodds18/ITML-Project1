############################# Imports #############################
import scipy as sp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

############################# Preprocessing the dataset #############################

# Import the dataset 
movies = pd.read_csv("tmdb_5000_movies.csv")

# Print instances 
countInstances = movies.count()
print(countInstances)

#Find attribute names
print("attribute names: %s" %  movies.columns) 

# Average, standard dev, median of values
print(movies.describe())

# Possible outcomes 
print(len(movies.revenue.unique()))

print(movies.spoken_languages.unique())


# Normalize the data, delete useless columns 
columnsToDrop = ['homepage', 'id', 'keywords','original_title', 'overview', 'popularity', 'release_date', 'status', 'tagline', 'title', 'vote_count']
movies.drop(columnsToDrop, inplace=True, axis=1)

# Print the columns after normalizing the data 
print("attribute names: %s" %  movies.columns) 


# Split the data into input and output 
array = movies.values
X = array[:,0:8]
Y = array[:,8]

# Change the output to classes from 1-10 
for y in range(0, len(Y)):
	if 0 < Y[y] <= 1:
		Y[y] = 1
	if 1 < Y[y] <= 2:
		Y[y] = 2
	if 2 < Y[y] <= 3:
		Y[y] = 3
	if 3 < Y[y] <= 4:
		Y[y] = 4
	if 4 < Y[y] <= 5:
		Y[y] = 5
	if 5 < Y[y] <= 6:
		Y[y] = 6
	if 6 < Y[y] <= 7:
		Y[y] = 7
	if 7 < Y[y] <= 8:
		Y[y] = 8
	if 8 < Y[y] <= 9:
		Y[y] = 9
	if 9 < Y[y] <= 10:
		Y[y] = 10

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=42)


############################# Other stuff #############################
'''
#K NEAREST NEIGHBOUR
n_neighbors = 53

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
listOfk = list(range(0,6000))
neigh = list(filter(lambda x: x % 10 != 0, listOfk))

# empty list that will hold cv scores
cv_scores = []

# Perform 10-fold cross validation
# This is done to find the best value for k 
for k in neigh:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(scores.mean())
    print(k)

# This is the best value for k, at least on the range that we tested
print(max(cv_scores))

# Create the graph to display the classification
for weights in ['uniform', 'distance']:
    # Create an instance of the Neighbours Classifier and fit the data
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_train, y_train)

    # Plot the decision boundary. Assign a colour to each point in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

# Print the plot
plt.show()
'''