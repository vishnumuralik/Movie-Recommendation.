
import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/vishn/Downloads/rotten_tomatoes_movie_reviews.csv')
# Visualize missing values
msno.matrix(df)
plt.title('Missing Value Matrix')
plt.show()
