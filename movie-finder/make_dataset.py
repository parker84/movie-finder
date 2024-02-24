import pandas as pd
import plotly.express as px

def prepare_movies_data(movies, ratings):
    ratings['rating_scaled'] = ratings['rating'] / 5
    avg_rating_scaled = ratings['rating_scaled'].mean()
    shrinkage_factor = 100

    avg_rating_per_movie = ratings.groupby('movieId').rating_scaled.aggregate(['mean', 'count', 'sum']).reset_index()
    avg_rating_per_movie.columns = ['movieId', 'avg_rating', 'num_ratings', 'total_ratings']
    avg_rating_per_movie['movieId'] = avg_rating_per_movie['movieId'].astype(int)
    avg_rating_per_movie['emp_bayes_rating'] = (
    ((avg_rating_per_movie['avg_rating'] * avg_rating_per_movie['num_ratings']) + (shrinkage_factor * avg_rating_scaled)) /
    (avg_rating_per_movie['num_ratings'] + shrinkage_factor)
)
    avg_rating_per_movie['smoothed_rating'] = [
    row['emp_bayes_rating'] if row['avg_rating'] > avg_rating_scaled # if the movie has an above average rating => use emp bayes smoothing
    else row['avg_rating'] # if its below average just use that average
    for ix, row in avg_rating_per_movie.iterrows()
]
    movies = movies.merge(avg_rating_per_movie, on='movieId')
    movies['title_and_stats'] = (
    movies['title']
    + ' - rating:' + movies['smoothed_rating'].round(2).astype(str)
    +  ' - num_ratings:' + movies['num_ratings'].astype(str)
)
    movies.to_csv('data/processed/movies.csv', index=False)
    return movies

def viz_rating_smoothing(movies):
    p = px.scatter(
    movies,
    x='avg_rating',
    y='emp_bayes_rating',
    color='num_ratings',
    hover_name='title'
)
    p.update_layout(
    title='Average Rating vs Empirical Bayes Rating',
    xaxis_title='Average Rating',
    yaxis_title='Empirical Bayes Rating'
)
    p.write_html('data/processed/avg_vs_emp_bayes_rating.html')

    p = px.scatter(
    movies,
    x='avg_rating',
    y='smoothed_rating',
    color='num_ratings',
    hover_name='title'
)
    p.update_layout(
    title='Average Rating vs Smoothed Emp Bayes Rating',
    xaxis_title='Average Rating',
    yaxis_title='Smoothed Empirical Bayes Rating'
)
    p.write_html('data/processed/avg_vs_smoothed_emp_bayes_rating.html')

def extract_genres(movies):
    genres = movies.genres.str.split('|', expand=True)
    genres = (
    genres.stack()
    .reset_index(level=1, drop=True)
    .value_counts()
    .reset_index()
)
    genres.columns=['genre', 'count']
    genres.to_csv('data/processed/genres.csv')

def sample_ratings(ratings):
    users_sampled = ratings.userId.drop_duplicates().sample(1000, replace=False, random_state=42)
    ratings_1k_users = ratings[ratings.userId.isin(users_sampled)]
    ratings_1k_users.to_csv('data/processed/ratings-1k-users.csv', index=False)

if __name__ == '__main__':
    movies = pd.read_csv('data/raw/ml-25m/movies.csv')
    ratings = pd.read_csv('data/raw/ml-25m/ratings.csv')
    movies = prepare_movies_data(movies, ratings)
    viz_rating_smoothing(movies)
    extract_genres(movies)
    sample_ratings(ratings)