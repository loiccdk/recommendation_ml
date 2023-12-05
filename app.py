###### UTILS ######

# Import necessary libraries
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import pandas as pd

# Load the dataset
df = pd.read_csv('song_dataset.csv').sample(frac=0.025)

# Dataframe that will be used for the recommendations
new_df = df

# List of songs id, title and artist, without duplicate, in alphabetical order (will be used for display)
songs = df[['song', 'title', 'artist_name']].drop_duplicates().sort_values(by=['title']).set_index('song').T.to_dict()

# Load the dataset for the surprise library
reader = Reader(rating_scale=(1, 100))
data = Dataset.load_from_df(df[['user', 'song', 'play_count']], reader)

# Split into Train / Test
trainset, testset = train_test_split(data, test_size=0.25)

# Build the recommendation model using collaborative filtering (KNNBasic)
model = KNNBasic(sim_options={'user_based': False})
model.fit(trainset)

# Function to get recommendations for 1 specific user in the database
def get_recommendations_user(user_id, num_recommendations=1):
    user_unseen_songs = new_df.loc[~new_df['song'].isin(new_df[new_df['user'] == user_id]['song'])]['song'].unique()
    predictions = [model.predict(user_id, song) for song in user_unseen_songs]
    recommendations = sorted(predictions, key=lambda x: x.est, reverse=True)[:num_recommendations]
    return recommendations

# Function to get recommendations based on a list of songs
def get_recommendations(songs, num_recommendation = 1):
    # Copy the database
    new_df = df

    # Add a record for every song in the list, attached to a new user with id = 1
    for song in songs:
        new_df = pd.concat([new_df,pd.DataFrame([{'user' : '1', 'song': song, 'play_count': 1}])], ignore_index=True)

    # Use the previous function to get recommendation for the user we just created
    return get_recommendations_user('1', num_recommendation)


###### APPLICATION ######
# Import libraries
from flask import Flask, render_template, request

# App config
app = Flask(__name__, template_folder='templates')
app.debug = True

# Starting page
@app.route('/', methods=['GET'])
def dropdown():
    # HTML Template for song selection
    return render_template('select.html', songs=songs)

# Recommendation page
@app.route('/recommendation', methods = ['GET', 'POST'])
def submitForm():
    # Get values from the song form
    selectValue = request.form.getlist('songs')

    # Compute the recommendations (id of the songs)
    recommendations = get_recommendations(selectValue, 10)

    # Get the songs corresponding to the ids
    results = []
    for rec in recommendations :
        results.append(songs[rec.iid])

    # HTML Template for results
    return render_template('results.html', results=results)

# MAIN
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
