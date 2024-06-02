import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

# Load the first 50,000 rows of the movie dataset
movie_data = pd.read_csv('/Users/vishn/Downloads/rotten_tomatoes_movies.csv', nrows=50000)

# Select relevant features for similarity calculation
features = ['audienceScore', 'tomatoMeter', 'runtimeMinutes']

# Fill missing values with zeros for simplicity
movie_data[features] = movie_data[features].fillna(0)

# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(movie_data[features])

# Compute cosine similarity between movies based on selected features
similarity_matrix = cosine_similarity(scaled_features)

# Convert similarity matrix to DataFrame for easier manipulation
similarity_df = pd.DataFrame(similarity_matrix, index=movie_data['title'], columns=movie_data['title'])

# Convert movie features to a sparse matrix
movie_features_sparse = csr_matrix(scaled_features)

# Check the shape of the sparse matrix
print("Shape of the sparse matrix:", movie_features_sparse.shape)

# Apply Singular Value Decomposition (SVD)
k = min(movie_features_sparse.shape) - 1  # Choose k such that it satisfies 0 < k < min(A.shape)
U, sigma, Vt = svds(movie_features_sparse, k=k)  # Choosing k for latent factors

# Convert sigma to diagonal matrix
sigma = np.diag(sigma)

# Make predictions for movie features
predicted_features = np.dot(np.dot(U, sigma), Vt)

# Convert predicted features to DataFrame
predicted_features_df = pd.DataFrame(predicted_features, index=movie_data['title'], columns=features)

def recommend_similar_movies(movie_title, num_recommendations=5):
    """
    Function to recommend similar movies based on item-based filtering.
    - movie_title: Title of the mo.sort_values(ascending=False)

    Args:

    # Exclude the input movie itself from recommendations
    similar_movies = similar_movies.drop(movie_title)vie for which recommendations are sought.
    - num_recommendations: Number of recommendations to return.

    Returns:
    - List of recommended movie titles.
    """
    # Retrieve similarity scores for the given movie
    movie_similarities = similarity_df.loc[movie_title]

    # Sort movies by similarity scores in descending order
    similar_movies = movie_similarities

    # Get top N recommended movies
    recommendations = similar_movies.head(num_recommendations)

    return recommendations


# Example usage:
movie_title = "The Dark Knight"
recommended_movies = recommend_similar_movies(movie_title)
print("Recommended movies for '{}':".format(movie_title))
print(recommended_movies)
