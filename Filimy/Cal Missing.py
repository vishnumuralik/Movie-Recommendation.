import pandas as pd

# Load your dataset
df = pd.read_csv('/Users/vishn/Downloads/rotten_tomatoes_movie_reviews.csv')  # Replace 'your_dataset.csv' with the path to your dataset

# Calculate the percentage of missing values for each variable
missing_percentage = df.isnull().mean() * 100

# Print the percentage of missing values for each variable
print("Percentage of missing values for each variable:")
print(missing_percentage)
