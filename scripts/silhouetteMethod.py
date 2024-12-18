import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

# Caricamento del dataset
feature_matrix = pd.read_csv('feature_matrix.csv', index_col=0)

silhouette_scores = []

print("Inizio calcolo per il Silhouette Score...")
for k in range(2, 11):  # Il silhouette score non è definito per k=1
    print(f"Calcolando K-Means per k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(feature_matrix)
    score = silhouette_score(feature_matrix, labels)
    silhouette_scores.append(score)

# Disegna il grafico
plt.figure(figsize=(10, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.title('Silhouette Score')
plt.xlabel('Numero di Cluster (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)

# Salva e mostra il grafico
output_file = "silhouette_score.png"
plt.savefig(output_file)
print(f"Grafico salvato come {output_file}")
plt.show()
