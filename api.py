from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from contextlib import asynccontextmanager

# --- 1. LE CONTRAT DE DONNÉES (Pydantic) ---
# Ce contrat sécurise l'API : il définit exactement ce que Streamlit a le droit d'envoyer.
class SearchRequest(BaseModel):
    query: str
    duration: Optional[str] = "Indifférent"
    actors: Optional[List[str]] = []
    top_n: int = 10

# Dictionnaire global pour maintenir les données lourdes en RAM
app_data = {}

# --- 2. GESTIONNAIRE DE DÉMARRAGE (Allocation Mémoire) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Démarrage du serveur API et allocation de la RAM...")
    
    # Chargement du fichier Pickle
    try:
        data = pd.read_pickle('Data_movie.pkl')
    except FileNotFoundError:
        print("❌ ERREUR CRITIQUE : Data_movie.pkl introuvable.")
        yield
        return

    # On remplace les NaN par des chaînes vides pour éviter que le JSON plante lors de l'envoi
    df = data['df'].fillna("")
    
    print("🧠 Chargement du modèle Sémantique (E5-Large)...")
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    
    # Mise en cache dans la RAM du serveur
    app_data['df'] = df
    app_data['emb_overview'] = data['emb_overview']
    app_data['emb_metadata'] = data['emb_metadata']
    app_data['emb_cast'] = data.get('emb_cast', None)
    app_data['model'] = model
    
    print("✅ API Prête ! En attente de requêtes sur le port 8000...")
    yield 
    
    print("🛑 Extinction du serveur API et libération de la mémoire...")
    app_data.clear()

# Initialisation de FastAPI
app = FastAPI(title="CinéFilm Backend IA", lifespan=lifespan)

# --- 3. LE POINT D'ACCÈS (ENDPOINT) ---
@app.post("/recommend")
def get_recommendations(req: SearchRequest):
    """Reçoit la requête, exécute l'algorithme E5, et renvoie un JSON propre."""
    t0 = time.time()
    
    df = app_data['df']
    model = app_data['model']
    emb_overview = app_data['emb_overview']
    emb_metadata = app_data['emb_metadata']
    emb_cast = app_data['emb_cast']

    # Vectorisation de la demande
    query_vec = model.encode(["query: " + req.query])

    # Calcul des scores sémantiques
    score_overview = cosine_similarity(query_vec, emb_overview)[0]
    score_metadata = cosine_similarity(query_vec, emb_metadata)[0]
    final_score = (score_overview * 0.6 + score_metadata * 0.4)

    # Bonus Casting
    if emb_cast is not None:
        score_cast = cosine_similarity(query_vec, emb_cast)[0]
        final_score = (final_score * 0.8) + (score_cast * 0.2)

    # Bonus TMDB
    if 'vote_average' in df.columns:
        vote_avg = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0).values
        final_score = (final_score * 0.9) + ((vote_avg / 10) * 0.1)

    # Malus pour résumés vides
    is_empty_overview = df['overview'].astype(str).str.len() < 10
    final_score = np.where(is_empty_overview, final_score - 2.0, final_score)

    # Filtres Durs
    keep_mask = np.ones(len(df), dtype=bool)

    if req.duration == "Court (< 1h40)":
        keep_mask &= (pd.to_numeric(df['runtime'], errors='coerce') < 100).fillna(False).values
    elif req.duration == "Moyen (1h40 - 2h20)":
        runtime = pd.to_numeric(df['runtime'], errors='coerce')
        keep_mask &= ((runtime >= 100) & (runtime <= 140)).fillna(False).values
    elif req.duration == "Long (> 2h20)":
        keep_mask &= (pd.to_numeric(df['runtime'], errors='coerce') > 140).fillna(False).values

    if req.actors:
        selected_actors = set(req.actors)
        def check_actor(movie_actors):
            if not isinstance(movie_actors, list):
                return False
            return bool(set(movie_actors) & selected_actors)
        keep_mask &= df['actors'].apply(check_actor).values

    final_score[~keep_mask] = -np.inf

    # Tri algorithmique (Top 10)
    sorted_indices = np.argsort(final_score)[::-1]
    top_indices = [idx for idx in sorted_indices if np.isfinite(final_score[idx])][:req.top_n]

    if not top_indices:
        return {"results": [], "time_sec": round(time.time() - t0, 3)}

    # Préparation du JSON à renvoyer au client
    results_list = []
    for idx in top_indices:
        movie_dict = df.iloc[idx].to_dict()
        # On injecte le score IA pour pouvoir l'afficher côté Streamlit !
        movie_dict['ai_score'] = float(final_score[idx])
        results_list.append(movie_dict)

    t1 = time.time()
    
    # FastAPI convertit automatiquement ce dictionnaire Python en flux réseau JSON
    return {
        "results": results_list,
        "time_sec": round(t1 - t0, 3)
    }