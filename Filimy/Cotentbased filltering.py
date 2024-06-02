import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset (first 50,000 rows)
movies = pd.read_csv('/Users/vishn/Downloads/rotten_tomatoes_movies.csv', nrows=50000)

# Preprocess text attributes (title, genre, director, writer)
movies['title'] = movies['title'].fillna('')
movies['genre'] = movies['genre'].fillna('')
movies['director'] = movies['director'].fillna('')
movies['writer'] = movies['writer'].fillna('')

# Combine text attributes into a single string
movies['combined_features'] = movies['title'] + ' ' + movies['genre'] + ' ' + movies['director'] + ' ' + movies['writer']

# Initialize TF-IDF vectorizer with sparse matrix output
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(movies['combined_features'])

# Calculate cosine similarity matrix using linear_kernel with sparse matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend movies based on similarity
def recommend_movies(movie_title, cosine_sim=cosine_sim, movies=movies):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude the movie itself
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Example usage: Recommend movies similar to 'The Dark Knight'
recommendations = recommend_movies('The Dark Knight')
print(recommendations)