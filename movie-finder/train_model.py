import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from scipy.sparse.linalg import svds
import numpy as np
import logging
import coloredlogs
from decouple import config
logger = logging.getLogger(__name__)
coloredlogs.install(level=config('LOG_LEVEL', 'INFO'))

ratings = pd.read_csv('data/processed/ratings-1k-users.csv')
movies = pd.read_csv('data/processed/movies.csv')
NEW_USER_ID = -1

def create_new_user_ratings(users_favourite_movies: list, movies: pd.DataFrame):
    new_user_ratings = pd.DataFrame({
        'userId': [NEW_USER_ID] * len(users_favourite_movies),
        'movieId': movies[movies.title.isin(users_favourite_movies)].movieId,
        'rating': [5] * len(users_favourite_movies),
        'rating_scaled': [1.0] * len(users_favourite_movies)
    })
    return new_user_ratings 

def train_model(data):
    algo = SVD()
    algo.fit(data.build_full_trainset())
    # cv = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    return algo

def get_estimated_movie_ratings(users_favourite_movies: list, training_ratings: pd.DataFrame):
    logger.info('Prepping the ratings data...')
    new_user_ratings = create_new_user_ratings(users_favourite_movies, movies)
    ratings = pd.concat([training_ratings, new_user_ratings], axis=0, ignore_index=True)
    # TODO: making a validation set and optimize k
    R_df = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    R = R_df.values
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    logger.info('Decomposing the matrix...')
    U, sigma, Vt = svds(R_demeaned, k = 50)
    logger.info('Reconstructing the matrix...')
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
    # reader = Reader(rating_scale=(0.5, 5))
    # data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    # algo = train_model(data)
    # estimated_ratings = []
    # import ipdb; ipdb.set_trace()
    # for movie in movies.movieId:
    #     estimated_ratings.append(algo.estimate(NEW_USER_ID, movie))
    movies['estimated_rating'] = preds_df.iloc[0]
    return movies.sort_values('estimated_rating', ascending=False)


if __name__ == '__main__':
    users_favourite_movies = [
        "Shawshank Redemption, The (1994)", 
        "Dark Knight, The (2008)",
        "Godfather: Part II, The (1974)",
        "Pulp Fiction (1994)",
        "Fight Club (1999)"
    ]
    logger.info(get_estimated_movie_ratings(users_favourite_movies, ratings))