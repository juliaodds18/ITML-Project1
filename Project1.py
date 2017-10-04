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
mushrooms = pd.read_csv("mushrooms.csv")


