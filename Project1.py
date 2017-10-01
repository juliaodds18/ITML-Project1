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

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.20, random_state=42)


print(len(y_train))
print(len(y_test))

############################# Other stuff #############################