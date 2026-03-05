import requests
import pandas as pd
import numpy as np
import pickle
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import concurrent.futures 

# --- 1. CONFIGURATION ---
TMDB_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJjMWMwMmZhOTkwNjQwMDhlMzg0ZWM0MTcxODEzZjc5NiIsIm5iZiI6MTc3MjQ3NzA5My4zNjA5OTk4LCJzdWIiOiI2OWE1ZGFhNWNmMWUwNzZlZDVjNGMwYTgiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.D3MM42AYCuhGlKkV_8vHpkIhbFBDTKUl9HltQX7B5Cs" # Pense à remettre ta clé !
BASE_URL = "https://api.themoviedb.org/3"   
HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_API_KEY}"
}

TARGET_MOVIES = 6000 
MIN_RATING = 5.0
CURRENT_YEAR = 2026

def get_movies_list():
    """Récupère une large liste de films populaires notés > 5/10."""
    movies = []
    for page in tqdm(range(1, 400), desc="Extraction TMDB (Métadonnées)"):
        url = f"{BASE_URL}/discover/movie?include_adult=false&include_video=false&language=fr-FR&page={page}&sort_by=popularity.desc&vote_average.gte={MIN_RATING}&vote_count.gte=100"
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code == 200:
            data = response.json()
            movies.extend(data.get('results', []))
        else:
            time.sleep(0.5) 
            
    return pd.DataFrame(movies)

def apply_age_penalty(df):
    """Applique un malus d'ancienneté et trie le dataset."""
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    df['release_year'] = df['release_year'].fillna(CURRENT_YEAR)
    df['age'] = CURRENT_YEAR - df['release_year']
    df['age_malus'] = df['age'].apply(lambda x: max(0, x) * 0.02)
    df['adjusted_score'] = df['vote_average'] - df['age_malus']
    df = df.sort_values(by='adjusted_score', ascending=False).head(TARGET_MOVIES).reset_index(drop=True)
    return df

def fetch_movie_details(movie_id):
    """Récupère le casting, les mots-clés, genres, tagline et trailer pour un film."""
    details = {
        'actors': [], 'director': '', 'keywords': [], 
        'genres': [], 'tagline': '', 'trailer_key': ''
    }
    
    def safe_get(url):
        for _ in range(3):
            res = requests.get(url, headers=HEADERS)
            if res.status_code == 200:
                return res.json()
            elif res.status_code == 429:
                time.sleep(0.5) 
        return {}

    # 1. Détails + Vidéos FR en UNE seule requête (Optimisation majeure)
    res_details = safe_get(f"{BASE_URL}/movie/{movie_id}?language=fr-FR&append_to_response=videos")
    
    details['runtime'] = res_details.get('runtime', 120)
    details['tagline'] = res_details.get('tagline', '')
    
    if 'genres' in res_details:
        details['genres'] = [g['name'] for g in res_details['genres']]

    # Extraction du Trailer FR
    if 'videos' in res_details and 'results' in res_details['videos']:
        for vid in res_details['videos']['results']:
            if vid.get('site') == 'YouTube' and vid.get('type') == 'Trailer':
                details['trailer_key'] = vid.get('key')
                break

    # Fallback : Si pas de trailer FR, on fait une petite requête pour le trailer US
    if not details['trailer_key']:
        res_videos_en = safe_get(f"{BASE_URL}/movie/{movie_id}/videos?language=en-US")
        if 'results' in res_videos_en:
            for vid in res_videos_en['results']:
                if vid.get('site') == 'YouTube' and vid.get('type') == 'Trailer':
                    details['trailer_key'] = vid.get('key')
                    break

    # 2. Casting & Réalisateur
    res_credits = safe_get(f"{BASE_URL}/movie/{movie_id}/credits?language=fr-FR")
    if 'cast' in res_credits:
        details['actors'] = [c['name'] for c in res_credits['cast'][:5]]
    if 'crew' in res_credits:
        directors = [c['name'] for c in res_credits['crew'] if c['job'] == 'Director']
        details['director'] = directors[0] if directors else ''
        
    # 3. Mots-clés
    res_kw = safe_get(f"{BASE_URL}/movie/{movie_id}/keywords")
    if 'keywords' in res_kw:
        details['keywords'] = [k['name'] for k in res_kw['keywords']]

    return details

def build_text_features(df):
    """Construit les chaînes de caractères qui seront vectorisées."""
    # Le modèle E5 fonctionne mieux avec le préfixe "passage: " pour les documents stockés
    prefix = "passage: "
    
    # On intègre la tagline au synopsis pour enrichir l'ambiance
    df['txt_overview'] = prefix + df['title'] + " : " + df['tagline'].fillna('') + " " + df['overview'].fillna('')
    
    # On gère proprement les listes pour les genres et les mots clés
    genres_str = df['genres'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    keywords_str = df['keywords'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    df['txt_metadata'] = prefix + genres_str + " " + keywords_str
    
    df['txt_cast'] = prefix + "Réalisateur: " + df['director'] + " Acteurs: " + df['actors'].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")
    return df

if __name__ == "__main__":
    print("🚀 Démarrage du Data Loader CinéFilm (Mode Multithread - E5 Large)...")
    
    df_raw = get_movies_list()
    df_raw = df_raw.drop_duplicates(subset=['id'])
    
    df_filtered = apply_age_penalty(df_raw)
    print(f"✅ {len(df_filtered)} films conservés après filtrage temporel.")
    
    print("⚡ Lancement de l'extraction API détaillée en parallèle...")
    movie_ids = df_filtered['id'].tolist()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        results = list(tqdm(executor.map(fetch_movie_details, movie_ids), total=len(movie_ids), desc="Détails API"))
    
    details_df = pd.DataFrame(results)
    df = pd.concat([df_filtered, details_df], axis=1)
    
    # --- PRÉPARATION DES DONNÉES FINALES ---
    df = build_text_features(df)
    
    df['poster_url'] = df['poster_path'].apply(
        lambda x: f"https://image.tmdb.org/t/p/w500{x}" if pd.notna(x) and str(x).strip() != "" else ""
    )
    
    print("🧠 Chargement du modèle E5-Large...")
    model = SentenceTransformer('intfloat/multilingual-e5-large')
    
    # Attention: E5-large est très lourd en VRAM. Si ton PC crash ici, descends le BATCH_SIZE à 32 ou 16.
    BATCH_SIZE = 64 
    
    print("⏳ Vectorisation des Synopsis (Overview + Tagline)...")
    emb_overview = model.encode(df['txt_overview'].tolist(), batch_size=BATCH_SIZE, show_progress_bar=True)
    
    print("⏳ Vectorisation des Métadonnées (Genres + Mots-clés)...")
    emb_metadata = model.encode(df['txt_metadata'].tolist(), batch_size=BATCH_SIZE, show_progress_bar=True)
    
    print("⏳ Vectorisation du Casting (Réalisateur/Acteurs)...")
    emb_cast = model.encode(df['txt_cast'].tolist(), batch_size=BATCH_SIZE, show_progress_bar=True)
    
    print("💾 Sauvegarde dans Data_movie.pkl...")
    
    # Liste finale des colonnes mise à jour avec les nouvelles données
    final_columns = [
        'id', 'title', 'release_date', 'runtime', 'vote_average', 
        'overview', 'tagline', 'genres', 'trailer_key', 'actors', 'director', 
        'txt_cast', 'txt_overview', 'txt_metadata', 'poster_url'
    ]
    
    export_data = {
        'df': df[final_columns],
        'emb_overview': emb_overview,
        'emb_metadata': emb_metadata,
        'emb_cast': emb_cast
    }
    
    with open('Data_movie.pkl', 'wb') as f:
        pickle.dump(export_data, f)
        
    print("🎉 Terminé ! Les données sont prêtes.")