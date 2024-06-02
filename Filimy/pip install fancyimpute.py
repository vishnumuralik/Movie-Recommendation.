import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('/Users/vishn/Downloads/rotten_tomatoes_movies.csv', nrows=50000)

# Check the proportion of missing values
missing_percentage = df.isnull().mean()

# Set a threshold for the proportion of missing values
threshold = 0.05  # For example, 5%

# Perform listwise deletion if the proportion of missing values is below the threshold
if missing_percentage.max() <= threshold:
    # Drop rows with any missing values
    df_cleaned = df.dropna()
    print("Listwise deletion performed. New dataset shape:", df_cleaned.shape)
else:
    print("Proportion of missing values exceeds the threshold. Listwise deletion not performed.")

# For numerical columns, perform mean imputation
numerical_columns_with_missing_values = df.select_dtypes(include=['float64', 'int64']).columns[df.isnull().any()]
mean_imputer = SimpleImputer(strategy='mean')
df[numerical_columns_with_missing_values] = mean_imputer.fit_transform(df[numerical_columns_with_missing_values])

# For categorical columns, perform mode imputation
categorical_columns_with_missing_values = df.select_dtypes(include=['object']).columns[df.isnull().any()]
mode_imputer = SimpleImpute
