###### UTILS

# Import necessary libraries
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import pandas as pd

# Load the dataset
df = pd.read_csv('song_dataset.csv').sample(frac=0.025)
new_df = df
# Load the dataset for the surprise library
reader = Reader(rating_scale=(1, 100))
data = Dataset.load_from_df(df[['user', 'song', 'play_count']], reader)

songs = df[['song', 'title', 'artist_name']].drop_duplicates().sort_values(by=['title']).set_index('song').T.to_dict()

trainset, testset = train_test_split(data, test_size=0.25)

# Build the recommendation model using collaborative filtering (KNNBasic)
model = KNNBasic(sim_options={'user_based': False})
model.fit(trainset)

# Function to get recommendations for a user
def get_recommendations_user(user_id, num_recommendations=1):
    user_unseen_songs = new_df.loc[~new_df['song'].isin(new_df[new_df['user'] == user_id]['song'])]['song'].unique()
    predictions = [model.predict(user_id, song) for song in user_unseen_songs]
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:num_recommendations]
    return recommendations

def get_recommendations(songs, num_recommendation = 1):
    new_df = df
    for song in songs:
        new_df = pd.concat([new_df,pd.DataFrame([{'user' : '1', 'song': song, 'play_count': 1}])], ignore_index=True)
    return get_recommendations_user('1', num_recommendation)


###### APPLICATION
from flask import Flask, render_template, request
app = Flask(__name__, template_folder='templates')
app.debug = True


@app.route('/', methods=['GET'])
def dropdown():
    return render_template('select.html', songs=songs)

@app.route('/submit-form', methods = ['GET', 'POST'])
def submitForm():
    selectValue = request.form.getlist('songs')
    recommendations = get_recommendations(selectValue, 20)
    results = []
    for rec in recommendations :
        results.append(songs[rec.iid])
    return render_template('results.html', results=results)

if __name__ == "__main__":
    app.run()
