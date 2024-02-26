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
    logger.info('Training the model...')
    svd = SVD()
    svd.fit(data.build_full_trainset())
    return svd

def eval_model(data):
    logger.info('Evaluating the model...')
    svd = SVD()
    cv = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    return cv

def get_estimated_movie_ratings(movies: pd.DataFrame, ratings: pd.DataFrame, users_favourite_movies: list, training_ratings: pd.DataFrame):
    logger.info('Prepping the ratings data...')
    new_user_ratings = create_new_user_ratings(users_favourite_movies, movies)
    ratings = pd.concat([training_ratings, new_user_ratings], axis=0, ignore_index=True)
    reader = Reader(rating_scale=(0.0, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    # eval_model(data) # MAE: 0.6568, RMSE: 0.8589
    svd = train_model(data)
    logger.info('Making predictions...')
    predictions = []
    for movie in movies.movieId.tolist():
        predictions.append(svd.predict(NEW_USER_ID, movie))
    logger.info('Cleaning up the predictions...')
    movies['predictions'] = predictions
    movies['estimated_rating'] = [pred.est for pred in predictions]
    movies['estimated_rating'] = movies['estimated_rating'] / 5.0
    users_fav_movies = movies[movies.title.isin(users_favourite_movies)]
    logger.info(f"Estimated ratings for user\'s favourite movies:\n{users_fav_movies[['title', 'estimated_rating']]}")
    movies = movies[movies.title.isin(users_favourite_movies) == False]
    movies = movies.sort_values('estimated_rating', ascending=False)
    logger.info('Done ✅')
    return movies


if __name__ == '__main__':
    users_favourite_movies = [
        "Shawshank Redemption, The (1994)", 
        "Dark Knight, The (2008)",
        "Godfather: Part II, The (1974)",
        "Pulp Fiction (1994)",
        "Fight Club (1999)"
    ]
    ratings = pd.read_csv('data/processed/ratings-1k-users.csv')
    movies = pd.read_csv('data/processed/movies.csv')
    logger.info(get_estimated_movie_ratings(movies, ratings, users_favourite_movies, ratings))