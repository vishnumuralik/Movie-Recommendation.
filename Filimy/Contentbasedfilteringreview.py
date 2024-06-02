import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset (first 50,000 rows)
df = pd.read_csv("/Users/vishn/Downloads/rotten_tomatoes_movie_reviews.csv", nrows=50000)  # Replace "your_dataset.csv" with the actual path to your dataset file

# Check if 'reviewText' column exists in the DataFrame
if 'reviewText' not in df.columns:
    print("Error: 'reviewText' column not found in the dataset.")
    exit()

# Preprocess text data
df['reviewText'] = df['reviewText'].fillna('')  # Fill missing values with empty string

# Compute TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['reviewText'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix)

# Function to recommend similar reviews
def recommend_similar_reviews(review_index, num_recommendations=5):
    sim_scores = list(enumerate(cosine_sim[review_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    review_indices = [i[0] for i in sim_scores]
    return df.iloc[review_indices]

# Example usage: Recommend similar reviews to the review at index 0
recommendations = recommend_similar_reviews(0)
print(recommendations[['reviewText', 'scoreSentiment']])
