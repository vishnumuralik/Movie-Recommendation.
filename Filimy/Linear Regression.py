import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('/Users/vishn/Downloads/rotten_tomatoes_movie_reviews.csv')

# Preprocess 'originalScore' column to extract numeric values
df['originalScore'] = df['originalScore'].str.extract(r'(\d+\.\d+)').astype(float)

# Identify and handle missing values
imputer = SimpleImputer(strategy='mean')
df['originalScore'] = imputer.fit_transform(df[['originalScore']])

# Drop rows with missing values in other columns
df.dropna(subset=['reviewText', 'reviewUrl'], inplace=True)

# Label encode the target variable 'scoreSentiment'
label_encoder = LabelEncoder()
df['scoreSentiment'] = label_encoder.fit_transform(df['scoreSentiment'])

# Extract relevant columns for analysis
X = df[['originalScore']]  # Features
y = df['scoreSentiment']    # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Print the coefficients
print("Coefficients:", model.coef_)

# Print the intercept
print("Intercept:", model.intercept_)
