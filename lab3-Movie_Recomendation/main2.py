import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def elbow_method(movie_gen_df):
    """Using the Elbow method to find the optimal number of clusters"""
    inertia = []
    k_range = range(1, 20)
    for k in k_range:
        kmean_model = KMeans(n_clusters=k, n_init=10)
        kmean_model.fit(movie_gen_df)
        inertia.append(kmean_model.inertia_)

    # Plotting the Elbow graph
    plt.figure(figsize=(16, 8))
    plt.plot(k_range, inertia, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


def find_similar_users(user_id, top_n=5):
    """ Function to find top N nearest neighbors for a given user """
    user_ratings = user_movie_matrix_scaled_df.loc[user_id, :].values.reshape(1, -1)
    user_ratings_df = pd.DataFrame(user_ratings, columns=user_movie_matrix_scaled_df.columns)
    distances, indices = knn.kneighbors(user_ratings_df, n_neighbors=top_n+1)
    similar_users = [(user_movie_matrix_scaled_df.index[indices.flatten()[i]], distances.flatten()[i]) for i in range(1, top_n+1)]
    return similar_users


def recommend_movies(user_id, similar_users, num_recommendations=5):
    """ Function to recommend movies for a user based on similar users' preferences """
    # Movies already rated by the user
    rated_movies = user_movie_matrix.loc[user_id]
    rated_movies = rated_movies[rated_movies > 0].index.tolist()

    # Aggregating movies rated by similar users
    recommended_movies = pd.DataFrame()
    for similar_user, _ in similar_users:
        similar_user_ratings = user_movie_matrix.loc[similar_user]
        recommended_movies = recommended_movies._append(similar_user_ratings)

    # Averaging the ratings and filtering out movies already rated by the user
    recommended_movies = recommended_movies.mean().sort_values(ascending=False).index.tolist()
    recommendations = [movie for movie in recommended_movies if movie not in rated_movies]

    return recommendations[:num_recommendations]


def anti_recommend_movies(user_id, similar_users, num_anti_recommendations=5):
    """Function to provide anti-recommendations for a user based on similar users' low ratings"""
    # Movies already rated by the user
    rated_movies = user_movie_matrix.loc[user_id]
    rated_movies = rated_movies[rated_movies > 0].index.tolist()

    # Aggregating movies rated by similar users
    anti_recommended_movies = pd.DataFrame()
    for similar_user, _ in similar_users:
        similar_user_ratings = user_movie_matrix.loc[similar_user]
        anti_recommended_movies = anti_recommended_movies._append(similar_user_ratings)

    # Averaging the ratings and filtering out movies already rated by the user
    anti_recommended_movies = anti_recommended_movies.mean().sort_values(ascending=True).index.tolist()
    anti_recommendations = [movie for movie in anti_recommended_movies if movie not in rated_movies]

    return anti_recommendations[:num_anti_recommendations]


# Load the provided CSV file
movies_file_path = 'movies.csv'
movies_df = pd.read_csv(movies_file_path)

# Load the users dataset
users_file_path = 'users.csv'
users_df = pd.read_csv(users_file_path)

# Load the ratings dataset
ratings_file_path = 'ratings.csv'
ratings_df = pd.read_csv(ratings_file_path)

# Merge the movies with the ratings
movies_ratings_df = pd.merge(movies_df, ratings_df, on='movie_id')

# Merge the resulting dataframe with the users data
combined_df = pd.merge(movies_ratings_df, users_df, on='user_id')

# One-hot encoding of the genres. Each genre is now a separate column with binary values (0 or 1).
genres_df = movies_df['genres'].str.get_dummies(sep='|')

# Creating a new dataframe with movie_id, title and the one-hot encoded genres
movies_with_genres_df = pd.concat([movies_df[['movie_id', 'title']], genres_df], axis=1)

# Plot to find the optimal number of clusters (for this set we decided for 8)
# elbow_method(genres_df)

# Cluster movies based on genres
kmeans = KMeans(n_clusters=8, n_init=10)
kmeans.fit(genres_df)

# Adding the cluster labels to the movies_with_genres_df
movies_with_genres_df['cluster'] = kmeans.labels_

# Creating a user-movie matrix with ratings
user_movie_matrix = combined_df.pivot_table(index='user_id', columns='title', values='rating')

# Replacing NaN values with 0, assuming that NaN means the user hasn't rated the movie
user_movie_matrix.fillna(0, inplace=True)

# Standardizing the user-movie matrix
scaler = StandardScaler()
user_movie_matrix_scaled = scaler.fit_transform(user_movie_matrix)

# Converting the scaled matrix back to a DataFrame
user_movie_matrix_scaled_df = pd.DataFrame(user_movie_matrix_scaled, index=user_movie_matrix.index, columns=user_movie_matrix.columns)

# Applying KNN using cosine similarity
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5, n_jobs=-1)
knn.fit(user_movie_matrix_scaled_df)


# Testing the function with a sample user (e.g., user_id = 1)
sample_user_id = 5
similar_users = find_similar_users(sample_user_id)

# Generating movie recommendations for the sample user
recommendations = recommend_movies(sample_user_id, similar_users)
print('Polecane filmy: ', recommendations)

# Generating anti-recommendations for the sample user
anti_recommendations = anti_recommend_movies(sample_user_id, similar_users)
print('Nie polecane filmy: ', anti_recommendations)
