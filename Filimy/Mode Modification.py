import pandas as pd

# Load your dataset
df = pd.read_csv('/Users/vishn/Downloads/rotten_tomatoes_movies.csv')  # Replace 'your_dataset.csv' with the path to your dataset

# Step 1: Identify categorical columns with missing values
categorical_columns_with_missing_values = df.select_dtypes(include=['object']).columns[df.select_dtypes(include=['object']).isnull().any()]

# Step 2: Calculate the mode of each categorical column
mode_values = df[categorical_columns_with_missing_values].mode().iloc[0]

# Step 3: Replace missing values in each categorical column with the mode
df_imputed = df.copy()  # Create a copy of the original dataframe to avoid modifying it directly
for column in categorical_columns_with_missing_values:
    # Replace missing values in each categorical column with the mode
    df_imputed[column] = df_imputed[column].fillna(mode_values[column])

# Display the first few rows of the imputed dataframe
print(df_imputed.head())
