import pandas as pd
import numpy as np
from scipy import stats

# Load your dataset
df = pd.read_csv('/Users/vishn/Downloads/rotten_tomatoes_movies.csv')  # Replace with the path to your dataset

# Calculate Z-scores
def calculate_z_scores(dataframe, columns):
    z_scores = np.abs(stats.zscore(dataframe[columns]))
    return z_scores

# Identify outliers using Z-scores
def identify_z_score_outliers(dataframe, columns, threshold=3):
    z_scores = calculate_z_scores(dataframe, columns)
    outliers = (z_scores > threshold).any(axis=1)
    return dataframe[outliers]

# Calculate IQR method
def calculate_iqr(dataframe, columns):
    Q1 = dataframe[columns].quantile(0.25)
    Q3 = dataframe[columns].quantile(0.75)
    IQR = Q3 - Q1
    return Q1, Q3, IQR

# Identify outliers using IQR method
def identify_iqr_outliers(dataframe, columns):
    Q1, Q3, IQR = calculate_iqr(dataframe, columns)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((dataframe[columns] < lower_bound) | (dataframe[columns] > upper_bound)).any(axis=1)
    return dataframe[outliers]

# Select numerical columns to analyze
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

# Identify outliers using Z-scores
z_score_outliers = identify_z_score_outliers(df, numerical_columns)
print(f"Outliers identified using Z-scores:\n{z_score_outliers}")

# Identify outliers using IQR method
iqr_outliers = identify_iqr_outliers(df, numerical_columns)
print(f"Outliers identified using IQR method:\n{iqr_outliers}")
