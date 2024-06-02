import pandas as pd

# Load your dataset
df = pd.read_csv('/Users/vishn/Downloads/rotten_tomatoes_movies.csv')  # Replace 'your_dataset.csv' with the path to your dataset

# Step 1: Identify numerical columns with missing values
numerical_columns_with_missing_values = df.select_dtypes(include=['float64', 'int64']).columns[df.select_dtypes(include=['float64', 'int64']).isnull().any()]

# Step 2: Calculate the mean or median of each numerical column
mean_values = df[numerical_columns_with_missing_values].mean()  # Change to .median() if you prefer median imputation

# Step 3: Replace missing values in each numerical column with the calculated mean or median
df_imputed = df.copy()  # Create a copy of the original dataframe to avoid modifying it directly
for column in numerical_columns_with_missing_values:
    # Replace missing values in each numerical column with the mean or median
    df_imputed[column] = df_imputed[column].fillna(mean_values[column])

# Display the first few rows of the imputed dataframe
print(df_imputed.head())
