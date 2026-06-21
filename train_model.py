import ast

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def classify_movie(row):
    if row['budget'] == 0:
        return 'Unknown'

    if row['revenue'] >= row['budget'] * 2:
        return 'Blockbuster'

    if row['revenue'] >= row['budget']:
        return 'Hit'

    return 'Flop'


def get_genre(value):
    try:
        parsed = ast.literal_eval(value)
        return parsed[0]['name'] if parsed else 'Unknown'
    except Exception:
        return 'Unknown'


def get_director(value):
    try:
        crew = ast.literal_eval(value)
        for person in crew:
            if person.get('job') == 'Director':
                return person.get('name', 'Unknown')
        return 'Unknown'
    except Exception:
        return 'Unknown'


def get_actor(value):
    try:
        cast = ast.literal_eval(value)
        return cast[0]['name'] if cast else 'Unknown'
    except Exception:
        return 'Unknown'


if __name__ == '__main__':
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')

    df = movies.merge(
        credits[['movie_id', 'cast', 'crew']],
        left_on='id',
        right_on='movie_id',
        how='left'
    )

    df = df[df['budget'] > 0].copy()
    df['success'] = df.apply(classify_movie, axis=1)
    df = df[df['success'] != 'Unknown'].copy()

    df['genre'] = df['genres'].apply(get_genre)
    df['director'] = df['crew'].apply(get_director)
    df['main_actor'] = df['cast'].apply(get_actor)

    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_month'] = df['release_date'].dt.month

    features = [
        'budget',
        'genre',
        'director',
        'cast',
        'crew',
        'main_actor',
        'popularity',
        'runtime',
        'vote_average',
        'vote_count',
        'release_month'
    ]

    X = df[features]
    y = df['success']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    num_cols = [
        'budget',
        'popularity',
        'runtime',
        'vote_average',
        'vote_count',
        'release_month'
    ]

    cat_cols = [
        'genre',
        'director',
        'main_actor'
    ]

    numeric_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),
            ('cat', categorical_transformer, cat_cols)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
        ]
    )

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, 'movie_success_model.pkl')
    print('Saved trained pipeline to movie_success_model.pkl')
