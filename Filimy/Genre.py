import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a function to convert boxOffice strings to numeric
def convert_to_numeric(value):
    if '$' in value:
        value = value.replace('$', '')  # Remove the '$' symbol
    if 'K' in value:
        return float(value.replace('K', '')) * 1000
    elif 'M' in value:
        return float(value.replace('M', '')) * 1000000
    else:
        return float(value)

# Load the dataset
df = pd.read_csv('/Users/vishn/Downloads/rotten_tomatoes_movieas.csv', nrows=50000)

# Drop rows with missing values in audienceScore, runtimeMinutes, and boxOffice columns
df_cleaned = df.dropna(subset=['audienceScore', 'runtimeMinutes', 'boxOffice'])

# Convert boxOffice column to numeric
df_cleaned['boxOffice'] = df_cleaned['boxOffice'].apply(convert_to_numeric)

# Select the variables for the 3D scatter plot
x = df_cleaned['audienceScore']
y = df_cleaned['runtimeMinutes']
z = df_cleaned['boxOffice']

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
ax.scatter(x, y, z, c='blue', marker='o')

# Set labels and title
ax.set_xlabel('Audience Score')
ax.set_ylabel('Runtime (minutes)')
ax.set_zlabel('Box Office Earnings (in millions)')
plt.title('3D Scatter Plot of Audience Score, Runtime, and Box Office Earnings')

# Show plot
plt.show()
