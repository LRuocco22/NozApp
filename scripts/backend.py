import pandas as pd
import pickle



# Settaggio dei dati per l'accesso a TMDb
TMDB_API_KEY = "89902622f730bc1e37ef7268bdebd0e0"
TMDB_BASE_URL = ""

# Selezione dei file del modello precedente generato che useremo per generare le raccomandazioni
movies_path = "processed_movies.csv"
model_path = "kmeans_model.pkl"
scaler_path = "scaler.pkl"

# Caricamento del modello e dello scaler
movies = pd.read_csv(movies_path)
with open(model_path, "rb") as model_file:
    kmeans_model = pickle.load(model_file)

with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)