import pandas as pd

# Percorsi dei file
movies_path = './movielens/movies.csv'
links_path = './movielens/links.csv'
genome_scores_path = './movielens/genome-scores.csv'
genome_tags_path = './movielens/genome-tags.csv'
ratings_path = './movielens/ratings.csv'

# Step 1: Caricamento dei file
movies_df = pd.read_csv(movies_path)
links_df = pd.read_csv(links_path)
genome_scores_df = pd.read_csv(genome_scores_path)
genome_tags_df = pd.read_csv(genome_tags_path)
ratings_df = pd.read_csv(ratings_path)

print("Tutti i file sono stati correttamente caricati")

ratings_info_by_id = ratings_df.groupby('movieId').agg(
    average_rating=('rating', 'mean'),  # Calcolo della valutazione media tra tutte le valutazioni dei diversi utenti
    rating_count=('rating', 'count')   # Numero di valutazioni per ogni film
)

movies_df = movies_df.merge(ratings_info_by_id, on='movieId', how='left') # effettuiamo il merge con join left su movieId

# generi in colonne binarie (one-hot encoding)
genres_split = movies_df['genres'].str.get_dummies('|')

#uniamo i generi binari con il dataframe movies. axis=1 significa unione per colonna
movies_df = pd.concat([movies_df, genres_split], axis=1)


# Riempimento dei valori mancanti nelle colonne average_rating e rating_count
movies_df['average_rating'] = movies_df['average_rating'].fillna(0)
movies_df['rating_count'] = movies_df['rating_count'].fillna(0)

# Estrazione dell'anno di rilascio dal titolo del film (es. "Movie Title (1995)")
movies_df['release_year'] = movies_df['title'].str.extract(r'\((\d{4})\)').astype(float)

# Riempimento degli anni mancanti con 0 e conversione a intero
movies_df['release_year'] = movies_df['release_year'].fillna(0).astype(int)

print("movie con valori mergati: ", movies_df.head(10))