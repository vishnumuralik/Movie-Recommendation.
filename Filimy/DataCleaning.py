#Data Cleaning for movie.csv
#Displaying all the missing values in movie.csv
import pandas as pd
dfm = pd.read_csv('/Users/vishn/Downloads/rotten_tomatoes_movies.csv')
dfm.isna().sum()
print(dfm)
#Data Cleaning for review.csv
#Displaying all the missing values in review.csv
import pandas as pd
dfr = pd.read_csv('/Users/vishn/Downloads/rotten_tomatoes_movie_reviews.csv')
dfr.isna().sum()
print(dfr)