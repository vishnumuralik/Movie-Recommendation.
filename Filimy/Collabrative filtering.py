

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

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


def recommend_similar_movies(movie_title, num_recommendations=5):
    """
    Function to recommend similar movies based on item-based filtering.

    Args:
    - movie_title: Title of the movie for which recommendations are sought.
    - num_recommendations: Number of recommendations to return.

    Returns:
    - List of recommended movie titles.
    """
    # Retrieve similarity scores for the given movie
    movie_similarities = similarity_df.loc[movie_title]

    # Sort movies by similarity scores in descending order
    similar_movies = movie_similarities.sort_values(ascending=False)

    # Exclude the input movie itself from recommendations
    similar_movies = similar_movies.drop(movie_title)

    # Get top N recommended movies
    recommendations = similar_movies.head(num_recommendations)

    return recommendations


# Example usage:
movie_title = "The Dark Knight"
recommended_movies = recommend_similar_movies(movie_title)
print("Recommended movies for '{}':".format(movie_title))
print(recommended_movies)
