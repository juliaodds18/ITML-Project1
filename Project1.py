import pandas as pd
movies = pd.read_csv("tmdb_5000_movies.csv")

# Print instances 
countInstances = movies.count()
print(countInstances)

#Find attribute names
print("attribute names: %s" %  movies)

