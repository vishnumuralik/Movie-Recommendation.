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

# Function to get recommendations based on a specified movie
def get_recommendations(movie_name, data, tfidf_matrix, cosine_sim):
    # Find the index of the specified movie
    movie_indices = data[data['title'].str.contains(movie_name, case=False, na=False)].index

    if len(movie_indices) == 0:
        return "No movie found with the specified name."

    # Take the first index if there are multiple matches
    movie_index = movie_indices[0]

    # Calculate similarity scores for the specified movie
    sim_scores = cosine_sim[movie_index, :]

    # Find the indices of the top 10 most similar movies
    similar_indices = sim_scores.argsort(axis=0)[-11:-1]

    # Return the titles of recommended movies
    return data['title'].iloc[similar_indices].tolist()

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

    # Example usage: recommend movies similar to a specified movie
    movie_name = input("Enter a movie name: ")
    recommendations = get_recommendations(movie_name, data, tfidf_matrix, cosine_sim)

    print("\nRecommended Movies:")
    for movie in recommendations:
        print(movie)
