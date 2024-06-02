#Here we are displaying all the information about movie.csv
import pandas as pd
dfm = pd.read_csv('/Users/vishn/Downloads/rotten_tomatoes_movies.csv')
print(dfm.info())
print(dfm.head(10))
print(dfm.tail(10))

#Here we are displaying all the information about review.csv
import pandas as pd
dfr = pd.read_csv('/Users/vishn/Downloads/rotten_tomatoes_movie_reviews.csv')
print(dfr.info())
print(dfr.head(10))
print(dfr.tail(10))


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('/Users/vishn/Downloads/rotten_tomatoes_movie_reviews.csv')

# Preprocess text data
df['processed_text'] = df['reviewText'].str.lower().str.replace(r'<.*?>', '', regex=True).str.replace(r'[^\w\s]', '', regex=True).fillna('')

# Compute TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])

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
