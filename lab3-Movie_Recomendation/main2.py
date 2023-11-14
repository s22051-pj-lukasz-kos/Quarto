import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the corrected users data
users_df_corrected = pd.read_csv('users.csv')

# Load the corrected movies data
movies_df_corrected = pd.read_csv('movies.csv')

# Load the corrected ratings data
ratings_df_corrected = pd.read_csv('ratings.csv')

# Merge the ratings with the movies
ratings_movies_df = pd.merge(ratings_df_corrected, movies_df_corrected, on='movie_id')

# Merge the resulting dataframe with the users
full_data_df = pd.merge(ratings_movies_df, users_df_corrected, on='user_id')

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Scale the 'rating' column
full_data_df['rating_scaled'] = scaler.fit_transform(full_data_df[['rating']])

# Create the user-item rating matrix for calculating the Pearson correlation
user_item_rating_matrix = full_data_df.pivot_table(index='user_id', columns='movie_id', values='rating_scaled')

# Fill missing values with zeros, which represent unrated movies
user_item_rating_matrix = user_item_rating_matrix.fillna(0)

# Calculate the Pearson correlation coefficient between users, not movies
user_similarity_matrix = user_item_rating_matrix.transpose().corr(method='pearson')


def recommend_movies(user_id, user_similarity_matrix, user_item_rating_matrix, n_top_users=5, n_recommendations=5):
    # Check if user_id exists in the similarity matrix
    if user_id not in user_similarity_matrix.index:
        raise ValueError(f"User ID {user_id} is not present in the user similarity matrix.")

    # Get the top n similar users
    similar_users = user_similarity_matrix[user_id].sort_values(ascending=False).head(n_top_users + 1)[1:]

    # Get the ratings of these users and multiply by the similarity scores
    similar_users_ratings = user_item_rating_matrix.loc[similar_users.index]
    similar_users_ratings = similar_users_ratings.multiply(similar_users.values, axis=0)

    # Get the sum of similarity scores for weighted average calculation
    sum_of_weights = similar_users_ratings.notnull().multiply(similar_users.values, axis=0).sum(axis=0)

    # Calculate the weighted average score
    movie_scores = similar_users_ratings.sum(axis=0) / sum_of_weights

    # Remove movies already rated by the user
    rated_movies = user_item_rating_matrix.loc[user_id, user_item_rating_matrix.loc[user_id, :] > 0].index
    movie_scores = movie_scores.drop(rated_movies)

    # Get the top n movie recommendations
    movie_recommendations = movie_scores.sort_values(ascending=False).head(n_recommendations).index.tolist()

    # Get movie titles using the movie index
    movie_titles = movies_df_corrected[movies_df_corrected['movie_id'].isin(movie_recommendations)]['title'].tolist()

    return movie_titles


def anti_recommend_movies(user_id, user_similarity_matrix, user_item_rating_matrix, n_least_similar_users=5,
                          n_recommendations=5):
    # Check if user_id exists in the similarity matrix
    if user_id not in user_similarity_matrix.index:
        raise ValueError(f"User ID {user_id} is not present in the user similarity matrix.")

    # Get the least similar users
    least_similar_users = user_similarity_matrix[user_id].sort_values().head(n_least_similar_users)

    # Get the ratings of these users and multiply by the negative similarity scores
    least_similar_users_ratings = user_item_rating_matrix.loc[least_similar_users.index]
    least_similar_users_ratings = least_similar_users_ratings.multiply(-1 * least_similar_users.values, axis=0)

    # Get the sum of negative similarity scores for weighted average calculation
    sum_of_weights = least_similar_users_ratings.notnull().multiply(-1 * least_similar_users.values, axis=0).sum(axis=0)

    # Calculate the weighted average score
    movie_scores = least_similar_users_ratings.sum(axis=0) / sum_of_weights

    # Remove movies already rated by the user and movies with no negative weights (to avoid division by zero)
    rated_movies = user_item_rating_matrix.loc[user_id, user_item_rating_matrix.loc[user_id, :] > 0].index
    movie_scores = movie_scores.drop(rated_movies)
    movie_scores = movie_scores[movie_scores > 0]  # Keep scores that are positive after inverting the sign

    # Get the top n movie anti-recommendations (lowest scores)
    movie_anti_recommendations = movie_scores.sort_values().head(n_recommendations).index.tolist()

    # Get movie titles using the movie index
    movie_titles = movies_df_corrected[movies_df_corrected['movie_id'].isin(movie_anti_recommendations)][
        'title'].tolist()

    return movie_titles


# Example usage:
# Recommend movies for user_id 0
print('Recommendations')
print(recommend_movies(6, user_similarity_matrix, user_item_rating_matrix, n_top_users=5, n_recommendations=5))

# Anti-recommend movies for user_id 0
print('Anti-recommendations')
print(anti_recommend_movies(6, user_similarity_matrix, user_item_rating_matrix, n_least_similar_users=5, n_recommendations=5))
