import pandas as pd
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.preprocessing import StandardScaler
import requests


# Settaggio dei dati per l'accesso a TMDb
TMDB_API_KEY = "89902622f730bc1e37ef7268bdebd0e0"
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Selezione dei file del modello precedente generato che useremo per generare le raccomandazioni
movies_path = "processed_movies.csv"
model_path = "kmeans_model.pkl"
scaler_path = "scaler.pkl" # Oggetto scaler per la normalizzazione dei dati

# Caricamento del modello e dello scaler
movies = pd.read_csv(movies_path)
with open(model_path, "rb") as model_file:
    kmeans_model = pickle.load(model_file)

with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file) 

#effetuiamo mapping tra i generi di tmdb e quelli presenti nel dataset
GENRE_MAPPING = {
    "Action": "Action",
    "Adventure": "Adventure",
    "Animation": "Animation",
    "Comedy": "Comedy",
    "Crime": "Crime",
    "Documentary": "Documentary",
    "Drama": "Drama",
    "Family": "Family",
    "Fantasy": "Fantasy",
    "History": "History",
    "Horror": "Horror",
    "Music": "Music",
    "Mystery": "Mystery",
    "Romance": "Romance",
    "Science Fiction": "Sci-Fi",
    "TV Movie": "TV Movie",
    "Thriller": "Thriller",
    "War": "War",
    "Western": "Western"
}

app = FastAPI()

class MovieRequest(BaseModel): #Movie Request è un oggetto che ha un unico campo 'tmdb_ids'
    tmdb_ids: List[int]  # Solo gli ID TMDb

def fetch_movie_details_from_tmdb(tmdb_id): # richiesta a TMDb per ottenere i dettagli del film
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
    params = {"api_key": TMDB_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except requests.exceptions.RequestException:
        return None

def fetch_movie_keywords_from_tmdb(tmdb_id): # chiamata per i tag in tmdb
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}/keywords"
    params = {"api_key": TMDB_API_KEY}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            keywords = response.json().get("keywords", [])
            return [kw["name"] for kw in keywords]
        else:
            return []
    except requests.exceptions.RequestException:
        return []

@app.post("/recommend")
def recommend_movies(payload: MovieRequest):
    
    
    tmdb_ids = payload.tmdb_ids
    preferred_tags = set()
    clusters = []  # Lista per memorizzare i cluster dei film dell'utente
    preferred_genres = set()

    print("Processing request with TMDb IDs: %s", tmdb_ids)

    # Ciclo sui film dell'utente per raccogliere generi, cluster e parole chiave
    for tmdb_id in tmdb_ids: # Se il film è presente nel dataset locale
        if tmdb_id in movies["tmdbId"].values:
            movie_row = movies[movies["tmdbId"] == tmdb_id].iloc[0]  # Ottiene la riga corrispondente (prendo solo i film che hanno la colonna tmdb = a tmdb_id)
            cluster = movie_row["cluster"] # prendo il cluster del film
            clusters.append(cluster)
            preferred_genres.update(movie_row["genres"].split('|'))
            preferred_tags.update(str(movie_row["tags"]).split('|'))
        else:
            # Se il film non è presente nel dataset, richiediamo i dettagli da TMDb
            movie_details = fetch_movie_details_from_tmdb(tmdb_id)
            if not movie_details:
                print("Failed to fetch details for TMDb ID: %s", tmdb_id)
                continue
            
            # Analizziamo i generi e li aggiungiamo all'insieme dei generi preferiti
            genres = movie_details.get("genres", [])
            for genre in genres:
                mapped_genre = GENRE_MAPPING.get(genre["name"])  # Mappiamo i generi TMDb con quelli locali
                if mapped_genre:
                    preferred_genres.add(mapped_genre)
            
             # Richiediamo le parole chiave del film in modo da conforntarli con i nostri tag successivamente
            keywords = fetch_movie_keywords_from_tmdb(tmdb_id)
            preferred_tags.update(keywords)  # Aggiunge le parole chiave ai tag preferiti
            
             # Costruisce il vettore delle feature del film
            genre_vector = [
                1 if col in preferred_genres else 0 for col in scaler.feature_names_in_
                if col not in ["average_rating", "rating_count"]
            ]
            avg_rating = movie_details.get("vote_average", 0) # Ottieniamo il voto medio
            rating_count = movie_details.get("vote_count", 0) # Ottieniamo il numero di voti
            feature_vector = [avg_rating, rating_count] + genre_vector

            # Scala il vettore e predice il cluster associato
            feature_vector_df = pd.DataFrame([feature_vector], columns=scaler.feature_names_in_)
            feature_vector_scaled = scaler.transform(feature_vector_df)
            cluster = kmeans_model.predict(feature_vector_scaled)[0]
            clusters.append(cluster)

    # Verifichiamo se sono stati trovati cluster o generi validi dai film forniti dall'utente.
    if not clusters or not preferred_genres:
        print("No valid clusters or genres found.")
        raise HTTPException(status_code=404, detail="No valid movies or genres found to generate recommendations.")

    # Determiniamo il cluster più frequente tra quelli associati ai film preferiti dall'utente.
    most_relevant_cluster = max(set(clusters), key=clusters.count)
    recommendations = movies[movies["cluster"] == most_relevant_cluster]

    #Calcolo della similarità tra i film
    recommendations["similarity"] = recommendations.apply(
        lambda row: (
            sum(1 for genre in row["genres"].split('|') if genre in preferred_genres) +
            sum(1 for tag in str(row["tags"]).split('|') if tag in preferred_tags)
        ),
        axis=1
    )
    #Ordiniamo i film raccomandati per priorità.
    recommendations = recommendations.sort_values(
        by=["similarity", "average_rating", "rating_count"], ascending=[False, False, False]
    )

    # filtriamo per tag e generi . i film devono avere almeno tre tag in comune o due generi in comune.
    recommendations = recommendations[
        recommendations["genres"].apply(
            lambda x: sum(1 for genre in x.split('|') if genre in preferred_genres) > 1
        ) & recommendations["tags"].apply(
            lambda x: sum(1 for tag in str(x).split('|') if tag in preferred_tags) > 2
        )
    ]

    # Se non ci sono film sufficienti, si prova a prendere quelli dello stesso cluster
    # Ogni riga del DataFrame recommendations avrà una nuova colonna similarity
    # che contiene un punteggio numerico che rappresenta quanto il film (la riga) è simile ai gusti dell'utente, in base ai generi e ai tag in comune.
    # Sommiamo la similarità con i tag e i generi preferiti per ogni riga del DataFrame recommendations
    if recommendations.empty:
        recommendations = movies[movies["cluster"] == most_relevant_cluster]
        recommendations["similarity"] = recommendations.apply(
            lambda row: (
                sum(1 for genre in row["genres"].split('|') if genre in preferred_genres) +
                sum(1 for tag in str(row["tags"]).split('|') if tag in preferred_tags)
            ),
            axis=1
        )
        # li ordiniamo di nuovo come prima
        recommendations = recommendations.sort_values(
            by=["similarity", "average_rating", "rating_count"], ascending=[False, False, False]
        )

    # rimpiazziamo valori inf o NaN con "Unknown"
    recommendations = recommendations.replace([float('inf'), float('-inf')], 0)
    recommendations = recommendations.fillna('Unknown')

    recommendations = recommendations.sort_values(by=["average_rating", "rating_count"], ascending=False)

    # i film con data successiva al 2015, prendiamo solo i primi 10
    recent_recommendations = recommendations[recommendations["release_year"] >= 2015]
    if len(recent_recommendations) > 0:
        recent_recommendations = recent_recommendations.sample(
            min(10, len(recent_recommendations)), replace=True, random_state=42
        )

    # prendiamo altri 5 films dalla cluster più vicina
    older_recommendations = recommendations[recommendations["release_year"] < 2015]
    if len(older_recommendations) > 0:
        older_recommendations = older_recommendations.sample(
            min(5, len(older_recommendations)), replace=True, random_state=42
        )

    recommendations = pd.concat([recent_recommendations, older_recommendations])

    # oggetto da restituire
    result = [
        {
            "tmdbId": movie["tmdbId"],
            "title": movie["title"],
            "genres": movie["genres"],
            "tags": "|".join(
                set(tag for tag in str(movie["tags"]).split('|') if tag in preferred_tags)
            )
        }
        for _, movie in recommendations.iterrows()
    ]

    if not result:
        raise HTTPException(status_code=404, detail="No recommendations found.")
    
    print("Final recommended movies: %d", len(result))
    return {"recommended_movies": result}