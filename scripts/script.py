import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

# Percorsi dei file
movies_path = '../movielens/movies.csv'
links_path = '../movielens/links.csv'
ratings_path = '../movielens/ratings.csv'
tags_path = '../movielens/tags.csv'

# Step 1: Caricamento dei file
movies_df = pd.read_csv(movies_path)
links_df = pd.read_csv(links_path)
ratings_df = pd.read_csv(ratings_path)
tags_df = pd.read_csv(tags_path)


ratings_info_by_id = ratings_df.groupby('movieId').agg(
    average_rating=('rating', 'mean'),  # Calcolo della valutazione media tra tutte le valutazioni dei diversi utenti
    rating_count=('rating', 'count')   # Numero di valutazioni per ogni film
)

movies_df = movies_df.merge(ratings_info_by_id, on='movieId', how='left') # effettuiamo il merge con join left su movieId

# generi in colonne binarie (one-hot encoding)
genres_split = movies_df['genres'].str.get_dummies('|')

#uniamo i generi binari con il dataframe movies. axis=1 significa unione per colonna
movies_df = pd.concat([movies_df, genres_split], axis=1)


tags_df['tag'] = tags_df['tag'].astype(str)
tags_aggregated = tags_df.groupby('movieId')['tag'].apply(lambda x: '|'.join(x)).reset_index()
tags_aggregated.rename(columns={'tag': 'tags'}, inplace=True)
movies_df = movies_df.merge(tags_aggregated, on='movieId', how='left')
movies_df['tags'] = movies_df['tags'].fillna('')

# Riempimento dei valori mancanti nelle colonne average_rating e rating_count
movies_df['average_rating'] = movies_df['average_rating'].fillna(0)
movies_df['rating_count'] = movies_df['rating_count'].fillna(0)

# Estrazione dell'anno di rilascio dal titolo del film (es. "Movie Title (1995)")
movies_df['release_year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)

# Riempimento degli anni mancanti con 0 e conversione a intero
movies_df['release_year'] = movies_df['release_year'].fillna(0).astype(int)

print("movie con valori mergati: ", movies_df.head(10))

# Applicazione di un peso alla variabile 'genres_split' (moltiplicandola per 'genre_weight')
# Si cerca di dare maggiore peso ai generi nel modello
genre_weight = 2
genres_split = genres_split * genre_weight

# Crea una lista di caratteristiche con 'average_rating', 'rating_count' e le colonne di 'genres_split'.
features = ['average_rating', 'rating_count'] + list(genres_split.columns)

# Estrazione del sottoinsieme delle caratteristiche selezionate da 'movies'
X = movies_df[features]

# Normalizzazione dei dati di 'X', con StandardScaler, per una maggiore comparabilità nel modello
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Esecuzione del clustering K-means con 'optimal_k' cluster sui dati standardizzati 'X_scaled'
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)

# Assegnamento dei cluster alle righe di 'movies'
movies_df['cluster'] = kmeans.labels_

# Unione dei dati con 'links' per aggiungere anche il 'tmdbId'
movies_df = movies_df.merge(links_df[['movieId', 'tmdbId']], on='movieId', how='left')

# # Salvataggio del modello KMeans e dello scaler in due file separati utilizzando pickle per la serializzazione
with open('kmeans_model.pkl', 'wb') as model_file:
    pickle.dump(kmeans, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)
    
# Salvataggio del DataFrame 'movies_df' in un file CSV
movies_df.to_csv('processed_movies.csv', index=False)

print("Il modello KMeans è stato salvato come 'kmeans_model.pkl'.")
print("Lo scaler è stato salvato come 'scaler.pkl'.")
print("Il DataFrame dei film è stato salvato come 'processed_movies.csv'.")


