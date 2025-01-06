**Obiettivo Del Progetto**
Il seguente progetto mira alla realizzazione di un modulo di raccomandazione, inserito in un'app per smart tv.
Il comportamento atteso è:
  1. Dare in input un array di id(film preferiti dall'utente);
  2. Ricevere una lista di film raccomandati che rispecchino le caratteristiche dei film passati in input;
Il progetto è stato realizzato utilizzando il dataset MovieLens, dal quale sono state considerate solo le feature utili al raggiungimento degli obiettivi del progetto.
Dopo una fase di Preprocessing dei dati sopracitati, si è analizzata la loro distribuzione tramite l'utilizzo di PCA.
Successivamente a questa analisi, si è deciso di affrontare il problema applicando tecniche di Clustering, virando verso la scelta dell'algoritmo KMeans.
Una volta applicato l'algoritmo di clustering, è stato realizzato un modulo di backend che interroga il dataset per la generazione della lista di film consigliati.
 
**Risorse Utilizzate**
1. Strumenti, Framework e Librerie
   1. Python: Linguaggio di programmazione usato per backend e script di elaborazione.
   2. FastAPI: Per la creazione di API backend.
   3. pydantic: Per la validazione dei dati in FastAPI.
   4. pandas: Per la manipolazione dei dati nei dataset.
   5. numpy: Per il calcolo numerico, ad esempio calcolo di distanze.
   6. scipy: Per calcolare distanze tra vettori (funzione cdist).
   7. sklearn (scikit-learn):
      1. StandardScaler: Per la normalizzazione dei dati.
      2. KMeans: Per il clustering dei dati.
      3. MultiLabelBinarizer: Per codificare i generi dei film in formato binario.
   8. FastAPI Middleware (CORS): Per abilitare richieste cross-origin. 
2. Dataset e File
   1. MovieLens dataset
   2. File generati:
      1. movies_with_clusters.csv: File che associa i film ai cluster.
      2. feature_matrix.csv: Matrice delle caratteristiche utilizzata per il clustering.
      3. kmeans_cluster_centers.csv: Centroidi calcolati per ogni cluster.

   
 
**Come replicare il progetto**
1. Visto la grande dimensione del dataset MovieLens, i file ".csv" sono stati esclusi dalla repository. Sono reperibili al seguente link: **"https://drive.google.com/drive/folders/1eDtbFOq_Yrr1O1fogWk21KBk8xzgGaTE?usp=sharing."**
2. I file scaricati dovranno essere inseriti nella cartella "movielens".
3. Accedere alla cartella "scripts".
4. Eseguire "py preprocessing.py" e attendere la sua esecuzione.
5. Controllare la corretta generazione dei file "feature_matrix.csv", "movies_with_clusters.csv", "kmeans_cluster_centres.csv".
6. Eseguire da terminale il seguente comando "uvicorn backend:app --reload" ed attendere fino alla configurazione completa del server.
7. All'interno del file "try_me.py" nella variabile "dati da inviare" inserire tra "[]" uno o più id presenti all'interno del file "movies_with_cluster.csv" separati da una virgola.
**Attenzione: inserire il tmdbId e non il movieID.** Esempi di input: "tmdb_ids": [277834]" , "tmdb_ids": [277834, 862]" , "tmdb_ids": [277834, 862, 354912]"
8. Eseguire da nuovo terminale "py try_me.py" e attendere la generazione dell'output.