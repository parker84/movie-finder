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
        '✅ Choose 1+ movies you like',
        options=movies.title_and_stats.tolist(),
        help='If you start typing the name of a movie (pre-2018), the list will be filtered accordingly for you to pick from.'
    )
    with st.expander('Filtering Options', expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            year_range = st.slider('Year Range', min_value=1900, max_value=2018, value=(1970, 2018))
        with col2:
            selected_genres = st.multiselect(
                'Genre(s)',
                genres.genre.tolist(),
                help='If you start typing the name of a genre, the list will be filtered accordingly for you to pick from.'
            )
    submit_button = st.form_submit_button(
        label='Find similar Movies 🎬',
        type='primary'
    )

    if submit_button:
        with st.spinner('Finding similar movies...'):
            estimated_ratings = get_estimated_movie_ratings(
                movies=movies,
                ratings=ratings,
                users_favourite_movies=movies[movies.title_and_stats.isin(selected_movies)].title.tolist(), 
                training_ratings=ratings
            )
            filtered_estimated_ratings = estimated_ratings[
                (estimated_ratings.year.between(year_range[0], year_range[1]))
            ]
            filtered_estimated_ratings = filtered_estimated_ratings[~filtered_estimated_ratings.title.isin(selected_movies)]
            if len(selected_genres) > 0:
                filtered_estimated_ratings['genre_isin'] = filtered_estimated_ratings['genres_list'].apply(lambda x: any([genre in x for genre in selected_genres]))
                filtered_estimated_ratings = filtered_estimated_ratings[filtered_estimated_ratings.genre_isin]
            filtered_estimated_ratings['your_rating'] = 100 * filtered_estimated_ratings['estimated_rating']
            filtered_estimated_ratings['avg_rating'] = 100 * filtered_estimated_ratings['smoothed_rating']
            filtered_estimated_ratings.index = range(1, filtered_estimated_ratings.shape[0]+1)
            st.dataframe(
                filtered_estimated_ratings[['title', 'your_rating', 'avg_rating', 'genres', 'num_ratings']].head(100),
                column_config={
                        "title": st.column_config.TextColumn(
                            "Title", width="large"
                        ),       
                        "your_rating": st.column_config.ProgressColumn(
                            "Rating (for you)",
                            format="%.0f%%", width="medium",
                            min_value=0,
                            max_value=100,
                        ),
                        "avg_rating": st.column_config.NumberColumn(
                            "Average Rating (all users)",
                            format="%.0f%%", width="medium",
                        ),
                        "num_ratings": st.column_config.NumberColumn(
                            "# of Ratings",
                            format="%.0f", width="medium",
                        ),
                        "genres": st.column_config.TextColumn(
                            "Genres", width="Large"
                        ),
                }
                        
            )