import pandas as pd 
import numpy as np
import operator
import random
import re 
from surprise import Reader, Dataset, SVD
from surprise.model_selection import KFold

movies_dataset=pd.read_csv('Dataset/ml-latest-small/movies.csv')
ratings_dataset=pd.read_csv('Dataset/ml-latest-small/ratings.csv')

''' Data Preprocessing '''
Genre=[]
Genres={}
for num in range(0,len(movies_dataset)):
    key=movies_dataset.iloc[num]['title']
    value=movies_dataset.iloc[num]['genres'].split('|')
    Genres[key]=value
    Genre.append(value)
    
movies_dataset['new'] = Genre

p = re.compile(r"(?:\((\d{4})\))?\s*$")

years=[]
for movies in movies_dataset['title']:
     m = p.search(movies)
     year = m.group(1)
     years.append(year)
movies_dataset['year']=years

movies_name=[]
raw=[]
for movies in movies_dataset['title']:
     m = p.search(movies)
     year = m.group(0)
     new=re.split(year,movies)
     raw.append(new)

for i in range(len(raw)):
    movies_name.append(raw[i][0][:-2].title())
    
movies_dataset['movie_name'] = movies_name
movies_dataset['new'] = movies_dataset['new'].apply(' '.join)

'''Applying the Cotent Based Filtering'''
# Applying Feature extraction 
from sklearn.feature_extraction.text import TfidfVectorizer

tfid = TfidfVectorizer(stop_words='english')
matrix = tfid.fit_transform(movies_dataset['new'])

# Compute the cosine similarity of every genre
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(matrix, matrix)

'''Applying the Collaborative Filtering'''
# Making the dataset containing the column as userid itemid ratings
# the order is very specific and we have to follow the same order
reader = Reader()
dataset = Dataset.load_from_df(
    ratings_dataset[['userId','movieId','rating']],
    reader
)

# Intialising the SVD model and specifying the number of latent features
# we can tune this parameters according to our requirement
svd=SVD(n_factors=25)

# evaluting the model on the based on the root mean square error and Mean absolute error 
# evaluate.evaluate(svd,dataset, measures=['rmse','mae'])

train = dataset.build_full_trainset()
svd.fit(train)

movies_dataset = movies_dataset.reset_index()
titles = movies_dataset['movie_name']
indices = pd.Series(
    movies_dataset.index,
    index=movies_dataset['movie_name']
)
ratings_dataset = ratings_dataset.drop(['timestamp'], axis=1).astype('int32')

def recommendation_content_based(user_id, input_movie):
    best_movie_id = input_movie
    result = {}
    sim_scores = list(enumerate(cosine_sim[best_movie_id]))

    sim_scores = sorted(
        sim_scores,
        key=lambda x:x[1],
        reverse=True
    )

    movie_id = [i[0] for i in sim_scores]

    for movie in movie_id:
        if best_movie_id == movie:
            continue
        
        ratings = ratings_dataset[ratings_dataset['movieId'] == movie]['rating']
        
        id_movies=movies_dataset[movies_dataset['movie_name']==titles[movie]]['movieId'].iloc[0]
        predicted_rating = round(svd.predict(user_id, movie).est, 2)

        result[movie] = predicted_rating

    return result

def recommendation_collaborative_based(user_id, input_movie):
    best_movie_id = input_movie
    sim_scores = list(enumerate(cosine_sim[best_movie_id]))

    sim_scores = sorted(
        sim_scores,
        key=lambda x:x[1],
        reverse=True
    )

    movie_id = [i[0] for i in sim_scores]

    ratings = {}
    for movie in movie_id:
        if best_movie_id == movie:
            continue

        ratings[movie] = round(svd.predict(user_id, movie).est, 2)
    
    return ratings

def ensemble_recommender(user_id=4):
    movies = ratings_dataset[ratings_dataset['userId'] == user_id]
    best_movie = movies.iloc[movies['rating'].idxmax()]
    best_movie_id = best_movie['movieId']

    collab_pred = recommendation_collaborative_based(user_id, best_movie_id)
    content_pred = recommendation_content_based(user_id, best_movie_id)
    
    avg = {}
    for mov in collab_pred:
        avg[mov] = (collab_pred[mov] + content_pred[mov]) / 2
    
    result = dict(sorted(avg.items(), key=operator.itemgetter(1), reverse=True)[:10])

    print("based on your rating of " + titles[best_movie_id])
    return result

result = ensemble_recommender()
for key in result:
    print(titles[key])








    

    
        
