from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

# Crea l'app FastAPI
app = FastAPI()

# Abilita il supporto CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permetti l'accesso da tutti gli origin (modifica se necessario)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carica i file generati
movies_with_clusters = pd.read_csv('movies_with_clusters.csv')
feature_matrix = pd.read_csv('feature_matrix.csv', index_col=0)

# Modello per la richiesta
class RecommendationRequest(BaseModel):
    tmdb_ids: List[int]  # Lista degli ID TMDb


@app.post('/recommend')
def recommend(request: RecommendationRequest):
    tmdb_ids = request.tmdb_ids

    # Filtra i movieId corrispondenti agli ID TMDb inviati
    movie_ids = movies_with_clusters[movies_with_clusters['tmdbId'].isin(tmdb_ids)]['movieId']
    
    # Seleziona le caratteristiche dei film
    input_features = feature_matrix.loc[movie_ids]
    
    # Calcola la media dei vettori di caratteristiche
    mean_features = input_features.mean(axis=0)
    
    # Calcola la distanza tra il vettore medio e tutti i film
    distances = feature_matrix.apply(lambda x: np.linalg.norm(x - mean_features), axis=1)
    
    # Ordina i film in base alla distanza e seleziona i più vicini
    recommended_indices = distances.nsmallest(10).index
    recommended_movies = movies_with_clusters.loc[movies_with_clusters['movieId'].isin(recommended_indices)]
    
    # Restituisci i risultati
    return recommended_movies[['tmdbId', 'title']].to_dict(orient='records')

