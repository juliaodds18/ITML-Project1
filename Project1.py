############################# Imports #############################
import scipy as sp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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