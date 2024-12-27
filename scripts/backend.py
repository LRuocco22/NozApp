from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Abilita CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carica i file generati
movies_with_clusters = pd.read_csv('movies_with_clusters.csv')
feature_matrix = pd.read_csv('feature_matrix.csv', index_col=0)

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

    # Calcola la distanza tra il vettore medio e tutti i film in modo vettoriale
    distances = cdist(mean_features, feature_matrix.values, metric='euclidean')[0]

    # Ordina i film in base alla distanza e seleziona i più vicini
    recommended_indices = np.argsort(distances)[:20]
    recommended_movies = movies_with_clusters.iloc[recommended_indices]

    # Restituisci i risultati
    return recommended_movies[['tmdbId', 'title']].to_dict(orient='records')
