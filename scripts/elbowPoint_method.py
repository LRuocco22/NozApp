import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

# Caricamento del dataset
feature_matrix = pd.read_csv('feature_matrix.csv', index_col=0)

# Elenco per memorizzare i valori di SSE (Sum of Squared Errors)
sse = []

print("Inizio calcolo per Elbow Point...")
for k in range(1, 11):
    print(f"Calcolando K-Means per k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(feature_matrix)
    sse.append(kmeans.inertia_)

# Disegna il grafico
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Point')
plt.xlabel('Numero di Cluster (k)')
plt.ylabel('SSE')
plt.grid(True)

# Salva e mostra il grafico
output_file = "elbow_point.png"
plt.savefig(output_file)
print(f"Grafico salvato come {output_file}")
plt.show()
