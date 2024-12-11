import pandas as pd

# Percorsi dei file
movies_path = 'movies.csv'
links_path = 'links.csv'
genome_scores_path = 'genome-scores.csv'
genome_tags_path = 'genome-tags.csv'
ratings_path = 'ratings.csv'

# Step 1: Caricamento dei file
movies_df = pd.read_csv(movies_path)
links_df = pd.read_csv(links_path)
genome_scores_df = pd.read_csv(genome_scores_path)
genome_tags_df = pd.read_csv(genome_tags_path)
ratings_path = pd.read_csv(ratings_path)

print("Tutti i file sono stati correttamente caricati")