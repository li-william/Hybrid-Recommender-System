from autoencoder import Autoencoder

import numpy as np
import pandas as pd
import pickle
import os.path

import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_movie_name(id_, movies_df):
    return movies_df.iloc[id_]['title']

class ContentRecommender(object):

    ''' Data proprocessing '''
    def __init__(self, data_folder):
        movies_df = pd.read_csv(data_folder + "/movies.csv")
        ratings_df = pd.read_csv(data_folder + "/ratings.csv").drop_duplicates('movieId')
        tags_df = pd.read_csv(data_folder+"/tags.csv")

        movies_df = pd.merge(movies_df, ratings_df, on="movieId", how="inner")
        movies_df.set_index('movieId', inplace=True)

        movies_df['genres'] = movies_df['genres'].str.replace(pat="|", repl=" ")
        movies_df['genres'] = movies_df['genres'].str.replace(pat="-", repl="")

        tags_df = pd.merge(tags_df, ratings_df, on="movieId", how="right")

        tags_df.fillna("", inplace=True)
        tags_df = pd.DataFrame(tags_df.groupby('movieId')['tag'].apply(lambda x: "{%s}" % ' '.join(x)))
        tags_df.reset_index(inplace=True)

        tags_df = pd.merge(movies_df, tags_df, left_index=True, right_on='movieId', how='right')
        tags_df['document'] = tags_df[['tag', 'genres']].apply(lambda x: ' '.join(x), axis=1)

        self.tags_df = tags_df
    
    ''' Helper function to TF-IDF vectorize prepreocessed data '''
    def _embed(self, min_df=0.0001):
        tfidf = TfidfVectorizer(
            ngram_range=(0, 1),
            min_df=min_df,
            stop_words='english'
        )
        tfidf_matrix = tfidf.fit_transform(self.tags_df['document'])

        self.tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            index=self.tags_df.index.tolist()
        )
        self.tfidf_df.to_pickle('./tfidf_matrix.pkl')
    
    ''' Training the Autoencoder on our TFIDF matrix '''
    def train(self, validation_perc=0.1, lr=1e-3, intermediate_size=500, encoded_size=100):
        if not os.path.isfile('tfidf_matrix.pkl'):
            self._embed(min_df=0.0001)
        else:
            with open('./tfidf_matrix.pkl', 'rb') as f:
                self.tfidf_df = pickle.load(f)
        
        ae = Autoencoder(
            self.tfidf_df,
            validation_perc=validation_perc,
            lr=lr,
            intermediate_size=intermediate_size,
            encoded_size=encoded_size,
        )

        ae.train_loop(epochs=40)
        losses = pd.DataFrame(data=list(zip(ae.train_losses, ae.val_losses)), columns=['train_loss', 'validation_loss'])
        losses['epoch'] = losses.index + 1

        self.losses = losses
        self.encoded_tfidf = ae.get_encoded_representations()

        with open('./autoencoder_embeddings.pkl', 'wb') as fh:
            pickle.dump(self.encoded_tfidf, fh)
        
        return self.encoded_tfidf

    ''' Calculate cosine similarity matrix on autoencoder embeddings '''
    def calculate_sim_matrix(self):
        if not os.path.isfile('autoencoder_embeddings.pkl'):
            self.train()
        else:
            with open('./autoencoder_embeddings.pkl', 'rb') as f:
                self.embeddings = pd.DataFrame(pickle.load(f))
        ids = list(self.embeddings.index)

        similarity_matrix = pd.DataFrame(cosine_similarity(
            X=self.embeddings),
            index=ids)
        similarity_matrix.columns = ids

        self.sim_matrix = similarity_matrix
        return self.sim_matrix

    ''' Make the top k recommendations for items similar to input '''
    def predict(self, item_index, k):
        if 'sim_matrix' not in dir(self):
            print("Can't predict until the similarity matrix has been calculated.")
            return
        
        result = pd.DataFrame(self.sim_matrix.loc[item_index])
        result.columns = ['score']
        result = result.sort_values('score', ascending=False)

        result = result.head(k)
        result.reset_index(inplace=True)
        result = result.rename(index=str, columns={"index": "item_id"})
        return result

    ''' Plot training and validation losses '''
    def plot_losses(self):
        if 'losses' not in dir(self):
            print("Can only plot losses after train() has been called")
            return

        plt.plot(self.losses['epoch'], self.losses['train_loss'])
        plt.plot(self.losses['epoch'], self.losses['validation_loss'])
        plt.ylabel('MSE loss')
        plt.xlabel('epoch')
        plt.show()

ID_TO_TEST = 1

movies_df = pd.read_csv("./Dataset/ml-latest-small/movies.csv").reset_index()

a = ContentRecommender("./Dataset/ml-latest-small")
sim = a.calculate_sim_matrix()
recommendations = a.predict(ID_TO_TEST, 10)

print("Recommendations for users who liked " + get_movie_name(ID_TO_TEST, movies_df))
for i, row in recommendations.iterrows():
    print(int(i)+1, get_movie_name(int(row['item_id']), movies_df))