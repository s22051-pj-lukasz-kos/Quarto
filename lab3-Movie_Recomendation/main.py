import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def find_neighborhood(user_id, n):
    model_knn = NearestNeighbors(metric="correlation", algorithm="brute")
    model_knn.fit(user_movies)
    distances, indices = model_knn.kneighbors(user_movies.iloc[user_id-1, :].values.reshape(1, -1), n_neighbors=n+1)
    similarities = 1-distances.flatten()
    print('{0} most similar users for user with id {1}:\n'.format(n, user_id))

    for i in range(0, len(indices.flatten())):
        # pomiń, jeśli ten sam użytkownik
        if indices.flatten()[i]+1 == user_id:
            continue
        else:
            print('{0}: User {1}, with similarity of {2}'.format(i, indices.flatten()[i]+1, similarities.flatten()[i]))
    return similarities, indices


def predict_rate(user_id, item_id, n):
    similarities, indices = find_neighborhood(user_id, n)
    neighborhood_ratings = []
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == user_id:
            continue
        else:
            neighborhood_ratings.append(user_movies.iloc[indices.flatten()[i],item_id-1])
    # delete weight for input user
    weights = np.delete(indices.flatten(), 0)
    prediction = round((neighborhood_ratings * weights).sum() / weights.sum())
    print('\nPredicted rating for user {0} -> item {1}: {2}'.format(user_id, item_id, prediction))


ratings_columns = ['movie_id', 'rating', 'user_id']
ratings = pd.read_csv('ratings.csv', names=ratings_columns)

user_movies = ratings.pivot(index='user_id', columns='movie_id', values="rating").reset_index(drop=True)
user_movies.fillna(0, inplace=True)
user_movies = pd.DataFrame(user_movies)

users_similarity = 1 - pairwise_distances(user_movies.to_numpy(), metric="correlation")
users_similarity_df = pd.DataFrame(users_similarity)

find_neighborhood(1, 10)
predict_rate(1, 11, 5)
