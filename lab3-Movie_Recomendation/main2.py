import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


class MovieRecommender:
    def __init__(self, movies_file_path, users_file_path, ratings_file_path):
        # Load datasets
        self.movies_df = pd.read_csv(movies_file_path)
        self.users_df = pd.read_csv(users_file_path)
        self.ratings_df = pd.read_csv(ratings_file_path)

        # Merge datasets
        self.movies_ratings_df = pd.merge(self.movies_df, self.ratings_df, on='movie_id')
        self.combined_df = pd.merge(self.movies_ratings_df, self.users_df, on='user_id')

        # One-hot encoding of genres
        genres_df = self.movies_df['genres'].str.get_dummies(sep='|')
        self.movies_with_genres_df = pd.concat([self.movies_df[['movie_id', 'title']], genres_df], axis=1)

        # Cluster movies based on genres
        self.cluster_movies()

        # Create user-movie matrix
        self.create_user_movie_matrix()

        # Standardize user-movie matrix
        self.standardize_user_movie_matrix()

        # Apply KNN using cosine similarity
        self.apply_knn()

    def cluster_movies(self, n_clusters=8):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        genres_df = self.movies_with_genres_df.drop(['movie_id', 'title'], axis=1)
        kmeans.fit(genres_df)
        self.movies_with_genres_df['cluster'] = kmeans.labels_

    def create_user_movie_matrix(self):
        self.user_movie_matrix = self.combined_df.pivot_table(index='user_id', columns='title', values='rating')
        self.user_movie_matrix.fillna(0, inplace=True)

    def standardize_user_movie_matrix(self):
        scaler = StandardScaler()
        self.user_movie_matrix_scaled = scaler.fit_transform(self.user_movie_matrix)
        self.user_movie_matrix_scaled_df = pd.DataFrame(self.user_movie_matrix_scaled,
                                                        index=self.user_movie_matrix.index,
                                                        columns=self.user_movie_matrix.columns)

    def apply_knn(self, n_neighbors=5):
        self.knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors, n_jobs=-1)
        self.knn.fit(self.user_movie_matrix_scaled_df)

    def elbow_method(self):
        inertia = []
        k_range = range(1, 20)
        for k in k_range:
            kmean_model = KMeans(n_clusters=k, n_init=10)
            kmean_model.fit(self.movies_with_genres_df.drop(['movie_id', 'title', 'cluster'], axis=1))
            inertia.append(kmean_model.inertia_)

        # Plotting the Elbow graph
        plt.figure(figsize=(16, 8))
        plt.plot(k_range, inertia, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('The Elbow Method showing the optimal k')
        plt.show()

    def find_similar_users(self, user_id, top_n=5):
        user_ratings = self.user_movie_matrix_scaled_df.loc[user_id, :].values.reshape(1, -1)
        user_ratings_df = pd.DataFrame(user_ratings, columns=self.user_movie_matrix_scaled_df.columns)
        distances, indices = self.knn.kneighbors(user_ratings_df, n_neighbors=top_n + 1)
        similar_users = [(self.user_movie_matrix_scaled_df.index[indices.flatten()[i]], distances.flatten()[i]) for i in
                         range(1, top_n + 1)]
        return similar_users

    def recommend_movies(self, user_id, similar_users, num_recommendations=5):
        rated_movies = self.user_movie_matrix.loc[user_id]
        rated_movies = rated_movies[rated_movies > 0].index.tolist()

        recommended_movies = pd.DataFrame()
        for similar_user, _ in similar_users:
            similar_user_ratings = self.user_movie_matrix.loc[similar_user]
            recommended_movies = recommended_movies._append(similar_user_ratings)

        recommended_movies = recommended_movies.mean().sort_values(ascending=False).index.tolist()
        recommendations = [movie for movie in recommended_movies if movie not in rated_movies]

        return recommendations[:num_recommendations]

    def anti_recommend_movies(self, user_id, similar_users, num_anti_recommendations=5):
        rated_movies = self.user_movie_matrix.loc[user_id]
        rated_movies = rated_movies[rated_movies > 0].index.tolist()

        anti_recommended_movies = pd.DataFrame()
        for similar_user, _ in similar_users:
            similar_user_ratings = self.user_movie_matrix.loc[similar_user]
            anti_recommended_movies = anti_recommended_movies._append(similar_user_ratings)

        anti_recommended_movies = anti_recommended_movies.mean().sort_values(ascending=True).index.tolist()
        anti_recommendations = [movie for movie in anti_recommended_movies if movie not in rated_movies]

        return anti_recommendations[:num_anti_recommendations]


# Example Usage
movies_file_path = 'movies.csv'
users_file_path = 'users.csv'
ratings_file_path = 'ratings.csv'

recommender = MovieRecommender(movies_file_path, users_file_path, ratings_file_path)

# Testing the function with a sample user (e.g., user_id = 1)
sample_user_id = 0
similar_users = recommender.find_similar_users(sample_user_id)

# Generating movie recommendations for the sample user
recommendations = recommender.recommend_movies(sample_user_id, similar_users)
print('Recommended movies: ', recommendations)

# Generating anti-recommendations for the sample user
anti_recommendations = recommender.anti_recommend_movies(sample_user_id, similar_users)
print('Not recommended movies: ', anti_recommendations)
