import streamlit as st
import pandas as pd
from train_model import get_estimated_movie_ratings


movies = pd.read_csv('data/processed/movies.csv')
genres = pd.read_csv('data/processed/genres.csv')
movies.sort_values('smoothed_rating', ascending=False, inplace=True)
ratings = pd.read_csv('data/processed/ratings-1k-users.csv')

st.set_page_config(
    page_title="Movie Finder",
    page_icon="📽️",
    layout="wide"
)
st.title('📽️ Movie Finder')
st.caption('Choose some of your favourite movies and we\'ll find similar ones that you\'ll like. Enjoy! 🍿') 

with st.form(key='my_form'):
    selected_movies = st.multiselect(
        '✅ Choose movies you like',
        options=movies.title_and_stats.tolist(),
        help='If you start typing the name of a movie (pre-2018), the list will be filtered accordingly for you to pick from.'
    )
    # TODO: remove the # of ratings from the selected movies options
    with st.expander('Filtering Options', expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            # TODO: change rating to a percent instead of a decimal (0.3 -> 30%)
            min_rating = st.slider('Minimum Rating', min_value=0.0, max_value=1.0, value=0.3, step=0.05)
            year_range = st.slider('Year Range', min_value=1900, max_value=2018, value=(1970, 2018))
            # TODO: min year filter isn't working 
        with col2:
            selected_genres = st.multiselect(
                'Filter by Genre',
                genres.genre.tolist(),
                help='If you start typing the name of a genre, the list will be filtered accordingly for you to pick from.'
            )
    submit_button = st.form_submit_button(
        label='Find similar Movies 🎬',
        type='primary'
    )

    if submit_button:
        # TODO: add a progress bar
        # TODO: add logging with timing
        estimated_ratings = get_estimated_movie_ratings(
            movies[movies.title_and_stats.isin(selected_movies)].title.tolist(), 
            training_ratings=ratings
        )
        filtered_estimated_ratings = estimated_ratings[
            (estimated_ratings.smoothed_rating >= min_rating) #&
            # (estimated_ratings.year.between(year_range[0], year_range[1]))
            # (estimated_ratings.genres.str.contains('|'.join(selected_genres)))
        ]
        # TODO: making smoothed_rating / estimated_rating on the output to be more user-friendly
        st.dataframe(filtered_estimated_ratings[['title', 'smoothed_rating', 'num_ratings', 'genres', 'estimated_rating']].head(25))