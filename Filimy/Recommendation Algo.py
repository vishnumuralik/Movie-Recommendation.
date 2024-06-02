import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# Read the CSV file into a DataFrame
def load_data(csv_file):
    return pd.read_csv(csv_file)


# Combine relevant features into a single text column for feature extraction
def combine_features(data):
    return data['title'].fillna('') + ' ' + data['genre'].fillna('') + ' ' + data['director'].fillna('')


# Initialize TF-IDF vectorizer with reduced features to reduce memory usage
def initialize_tfidf(stop_words='english', max_features=1000):
    return TfidfVectorizer(stop_words=stop_words, max_features=max_features)


# Fit TF-IDF vectorizer to the data
def fit_tfidf(tfidf, features):
    return tfidf.fit_transform(features)


# Compute cosine similarity
def compute_cosine_similarity(tfidf_matrix):
    return linear_kernel(tfidf_matrix, tfidf_matrix)


# Function to get recommendations based on a specified theme
def get_recommendations(theme, data, tfidf_matrix, cosine_sim):
    # Find the indices of the movies with the specified theme
    theme_indices = data[data['features'].str.contains(theme, case=False)].index

    if len(theme_indices) == 0:
        return "No movies found for the specified theme."

    # Calculate similarity scores for all movies with the specified theme
    sim_scores = cosine_sim[theme_indices, :]

    # Find the indices of the top 10 most similar movies
    similar_indices = sim_scores.argsort(axis=1)[:, -10:]
    num_similar_movies = min(10, len(theme_indices))

    # Get movie indices
    movie_indices = theme_indices[-num_similar_movies:]

    # Return the titles of recommended movies
    return data['title'].iloc[movie_indices].tolist()


if __name__ == "__main__":
    # Path to the CSV file
    csv_file_path = "/Users/vishn/Downloads/rotten_tomatoes_movies.csv"

    # Load data
    data = load_data(csv_file_path)

    # Select the first 50,000 entries
    data = data.head(50000)

    # Combine features
    data['features'] = combine_features(data)

    # Initialize TF-IDF vectorizer
    tfidf = initialize_tfidf()

    # Fit TF-IDF vectorizer to the data
    tfidf_matrix = fit_tfidf(tfidf, data['features'])

    # Compute cosine similarity
    cosine_sim = compute_cosine_similarity(tfidf_matrix)

    # Example usage: recommend movies based on a specified theme
    theme = input("Enter a theme: ")
    recommendations = get_recommendations(theme, data, tfidf_matrix, cosine_sim)

    print("\nRecommended Movies:")
    for movie in recommendations:
        print(movie)
