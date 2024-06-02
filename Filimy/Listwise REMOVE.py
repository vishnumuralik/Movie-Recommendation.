import pandas as pd

# Load your dataset
df = pd.read_csv('/Users/vishn/Downloads/rotten_tomatoes_movies.csv')  # Replace 'your_dataset.csv' with the path to your dataset

# Step 1: Calculate the percentage of missing values for each column
missing_percentage = df.isnull().mean()

# Step 2: Set a threshold for the percentage of missing values
threshold = 0.5  # For example, 50%

# Step 3: Identify columns with missing values exceeding the threshold
columns_to_remove = missing_percentage[missing_percentage > threshold].index

# Step 4: Remove identified columns from the dataset
df_cleaned = df.drop(columns=columns_to_remove)

print("Columns removed:", columns_to_remove)
print("New dataset shape:", df_cleaned.shape)
