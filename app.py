from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("movie_success_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    budget = float(request.form['budget'])
    genre = request.form['genre']
    director = request.form['director']
    cast = request.form['cast']
    crew = request.form['crew']
    main_actor = request.form['main_actor']
    popularity = float(request.form['popularity'])
    runtime = float(request.form['runtime'])
    vote_average = float(request.form['vote_average'])
    vote_count = float(request.form['vote_count'])
    release_month = int(request.form['release_month'])

    data = pd.DataFrame({
        'budget':[budget],
        'genre':[genre],
        'director':[director],
        'cast':[cast],
        'crew':[crew],
        'main_actor':[main_actor],
        'popularity':[popularity],
        'runtime':[runtime],
        'vote_average':[vote_average],
        'vote_count':[vote_count],
        'release_month':[release_month]
    })

    prediction = model.predict(data)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Movie Success: {prediction}"
    )

if __name__ == "__main__":
    app.run(debug=True)