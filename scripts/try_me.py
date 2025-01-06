import requests

def effettua_richiesta_post(url, dati):
    try:
        print(f"Inviando dati a {url}: {dati}")
        risposta = requests.post(url, json=dati)
        
        if risposta.status_code == 200:
            print("Richiesta completata con successo!")
            dati_risposta = risposta.json()
            
            # Stampa ogni film su una linea separata
            for film in dati_risposta:
                print(f"- ID: {film['tmdbId']}, Titolo: {film['title']}, Cluster: {film['cluster']}, Generi: {film['genres']}")
        else:
            print(f"Errore nella richiesta: {risposta.status_code}")
            print("Dettagli:", risposta.text)
    except requests.exceptions.RequestException as e:
        print(f"Errore durante la richiesta: {e}")

url_endpoint = "http://127.0.0.1:8000/recommend"

dati_da_inviare = {
    "tmdb_ids": [277834, 862, 354912, 672] # TITOLI DEI FILM: MOANA, TOY STORY, COCO, HARRY POTTER E LA CAMERA DEI SEGRETI
}

effettua_richiesta_post(url_endpoint, dati_da_inviare)
