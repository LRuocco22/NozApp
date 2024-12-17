import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer

# Percorsi dei file
movies_path = '../movielens/movies.csv'
links_path = '../movielens/links.csv'
genome_scores_path = '../movielens/genome-scores.csv'
genome_tags_path = '../movielens/genome-tags.csv'

# Step 1: Caricamento dei file
movies_df = pd.read_csv(movies_path)
links_df = pd.read_csv(links_path)
genome_scores_df = pd.read_csv(genome_scores_path)
genome_tags_df = pd.read_csv(genome_tags_path)


# Step 2: Unione dei dataset
genome_scores_with_tags = genome_scores_df.merge(genome_tags_df, on="tagId")
movies_with_tags = movies_df.merge(genome_scores_with_tags, on="movieId")
final_dataset = movies_with_tags.merge(links_df[['movieId', 'tmdbId']], on="movieId")

# Step 3: Prepara i dati per clustering
movies_df['genres_list'] = movies_df['genres'].apply(lambda x: x.split('|'))
mlb = MultiLabelBinarizer()
genres_encoded = pd.DataFrame(mlb.fit_transform(movies_df['genres_list']),
                              columns=mlb.classes_, index=movies_df['movieId'])

# Media dei punteggi di rilevanza per tag
tag_relevance = final_dataset.groupby(['movieId', 'tag'])['relevance'].mean().unstack(fill_value=0)

# Step 4: Concatenazione generi e tag
feature_matrix = pd.concat([genres_encoded, tag_relevance], axis=1, sort=False).fillna(0)
feature_matrix = feature_matrix.loc[feature_matrix.index.intersection(movies_df['movieId'])]

# Step 5: Clustering con KMeans
num_clusters = 10
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(feature_matrix)

# Associare i cluster ai film
movies_with_clusters = movies_df.set_index('movieId').join(links_df.set_index('movieId'), how='inner')
movies_with_clusters['cluster'] = pd.Series(clusters, index=feature_matrix.index)

# Step 6: Salvataggio dei risultati
movies_with_clusters.reset_index().to_csv('movies_with_clusters.csv', index=False)
feature_matrix.to_csv('feature_matrix.csv', index=True)

print("Clustering completato. I risultati sono stati salvati.")

