from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

# Abilita CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# URL remoti dei file CSV
URL_MOVIES = "https://drive.google.com/file/d/1SBmmROivZtFDeFdozoChXNWrIxZvLzrV/view?usp=drive_link"
URL_FEATURES = "https://drive.google.com/file/d/1vEKewdJvJ3fItcTlIx_3TcJonFpufCn8/view?usp=drive_link"
URL_CENTROIDS = "https://drive.google.com/file/d/1AQ4IeSc_STefqFlmYkBTsU7q25Kvh2LK/view?usp=drive_link"

# Funzione per scaricare i file CSV
def download_csv(url):
    response = requests.get(url)
    response.raise_for_status()  # Genera un errore se il download fallisce
    return pd.read_csv(pd.compat.StringIO(response.text))

# Scarica i file CSV
movies_with_clusters = download_csv(URL_MOVIES)
feature_matrix = download_csv(URL_FEATURES)
kmeans_centroids = download_csv(URL_CENTROIDS).values

print("File CSV scaricati con successo.")

# Modello per l'input
class RecommendRequest(BaseModel):
    tmdb_ids: List[int]

@app.post("/recommend")
def recommend(request: RecommendRequest):
    tmdb_ids = request.tmdb_ids

    # Filtra i movieId corrispondenti agli ID TMDb inviati
    movie_ids = movies_with_clusters[movies_with_clusters['tmdbId'].isin(tmdb_ids)]['movieId']

    if movie_ids.empty:
        return ["Non ci sono film da Raccomandare all'utente"]  # Nessun film corrispondente

    # Seleziona le caratteristiche dei film
    input_features = feature_matrix.loc[movie_ids]

    # Calcola la media dei vettori di caratteristiche
    mean_features = input_features.mean(axis=0).values.reshape(1, -1)

    # Identifica il cluster più vicino al vettore medio
    closest_cluster = np.argmin(cdist(mean_features, kmeans_centroids, metric='euclidean'))

    # Filtra i film appartenenti al cluster identificato
    cluster_movies = movies_with_clusters[movies_with_clusters['cluster'] == closest_cluster]
    cluster_feature_matrix = feature_matrix.loc[cluster_movies['movieId']]

    # Calcola la distanza tra il vettore medio e i film filtrati
    distances = cdist(mean_features, cluster_feature_matrix.values, metric='euclidean')[0]

    # Ordina i film in base alla distanza e seleziona i più vicini
    recommended_indices = np.argsort(distances)[:20]
    recommended_movies = cluster_movies.iloc[recommended_indices]

    # Escludi i film passati nella richiesta
    recommended_movies = recommended_movies[~recommended_movies['tmdbId'].isin(tmdb_ids)]

    # Restituisci i risultati
    return recommended_movies[['tmdbId', 'title', 'cluster', 'genres']].to_dict(orient='records')
