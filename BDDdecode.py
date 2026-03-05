import time
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

print("🔄 Lancement de Streamlit...")

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="CinéFilm", layout="wide", page_icon="🎬")

# --- 2. CHARGEMENT DES DONNÉES ET DES MODÈLES ---
@st.cache_data
def load_data():
    """Charge le dictionnaire contenant le DataFrame et les matrices de vecteurs."""
    try:
        return pd.read_pickle('Data_movie.pkl')
    except FileNotFoundError:
        return None

#@st.cache_resource
#def load_cross_encoder():
#   return CrossEncoder('BAAI/bge-reranker-v2-m3')

@st.cache_resource
def load_model():
    # Modèle lourd pour la recherche vectorielle globale
    return SentenceTransformer('intfloat/multilingual-e5-large')
 #On charge Les modèles et la data
data = load_data()
model = load_model()
#cross_model = load_cross_encoder()

if data is None:
    st.error("⚠️ Fichier de données introuvable ! Vérifiez que 'Data_movie.pkl' est bien dans le dossier.")
    st.stop()

# Extraction des données
df = data['df']
emb_overview = data['emb_overview']
emb_metadata = data['emb_metadata']
emb_cast = data.get('emb_cast', None) 
print(type(df), df.shape, type(emb_overview), emb_overview.shape)
# --- 3. PRÉPARATION DES FILTRES ---
all_actors = set()
if 'actors' in df.columns:
    for actor_list in df['actors']:
        if isinstance(actor_list, list):
            for actor in actor_list:
                all_actors.add(actor)
sorted_actors = sorted(list(all_actors))
print(f"🎭 Acteurs/Réalisateurs uniques trouvés : {len(sorted_actors)}")


# --- 4. MOTEUR DE RECOMMANDATION ---
def recommend_movies(query, filters, top_n=10):
    """Calcule les recommandations en appliquant les filtres stricts et le reranking."""
    t0 = time.time()
    
    # Vectorisation de la demande
    query_vec = model.encode(["query: " + query])

    # Calcul des scores sémantiques (E5-Large)
    score_overview = cosine_similarity(query_vec, emb_overview)[0]
    score_metadata = cosine_similarity(query_vec, emb_metadata)[0]
    final_score = (score_overview * 0.5 + score_metadata * 0.5)

    # Bonus Casting
    if emb_cast is not None:
        score_cast = cosine_similarity(query_vec, emb_cast)[0]
        final_score = (final_score * 0.8) + (score_cast * 0.2)
    else:
        print("⚠️ Avertissement : Embeddings de casting non disponibles, ce critère sera ignoré.")
    # Bonus Note TMDB
    if 'vote_average' in df.columns:
        final_score = (final_score * 0.9) + ((df['vote_average'].fillna(0).values / 10) * 0.15)
        
    # Malus pour résumés vides (Correction Bug "Nude")
    is_empty_overview = df['overview'].fillna("").str.len() < 10
    final_score = np.where(is_empty_overview, final_score - 2.0, final_score)
    
    # --- APPLICATION DES FILTRES DURS ---
    keep_mask = np.ones(len(df), dtype=bool)

    if filters.get('duration') == "Court (< 1h40)":
        keep_mask &= (df['runtime'] < 100)
    elif filters.get('duration') == "Moyen (1h40 - 2h20)":
        keep_mask &= (df['runtime'] >= 100) & (df['runtime'] <= 140)
    elif filters.get('duration') == "Long (> 2h20)":
        keep_mask &= (df['runtime'] > 140)

    if filters.get('actors'):
        selected_actors = filters['actors']
        def check_actor(movie_actors):
            if not isinstance(movie_actors, list):
                return False
            return bool(set(movie_actors) & set(selected_actors))
        keep_mask &= df['actors'].apply(check_actor).values

    final_score[~keep_mask] = -np.inf

    # --- ÉTAPE UNIQUE : RETRIEVAL (Top 10 E5) ---
    sorted_indices = np.argsort(final_score)[::-1]
    
    # On garde les 10 meilleurs
    top_10_indices = [idx for idx in sorted_indices if np.isfinite(final_score[idx])][:top_n]
    
    if not top_10_indices:
        return pd.DataFrame()

    t1 = time.time()

# --- 🔍 DEBUG : LOGS DANS LE TERMINAL AVEC LE SCORE IA ---
    print("\n" + "="*50)
    print("🏆 CLASSEMENT FINAL (E5-Large Uniquement)")
    print("="*50)
    
    for i, idx in enumerate(top_10_indices):
        movie_title = df.iloc[idx]['title']
        # On récupère le score mathématique exact calculé par ton algorithme
        ai_score = final_score[idx] 
        
        print(f"  {i+1}. {movie_title} (Score IA : {ai_score:.4f})")
        
    print("="*50)
    print(f"⚡ Temps total de recherche : {t1 - t0:.3f} secondes\n")
    
    return df.iloc[top_10_indices]

# --- 5. GESTION DE L'ÉTAT (SESSION STATE) ---
if 'num_visible' not in st.session_state:
    st.session_state.num_visible = 5
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'exec_time' not in st.session_state:
    st.session_state.exec_time = 0.0

# --- 6. INTERFACE UTILISATEUR ---
st.title("🎬 CinéFilm")
st.markdown("Trouvez le film parfait grâce à l'IA et filtrez selon vos besoins.")
st.caption(f"📊 Base de données actuelle : {len(df)} films analysés par l'IA.")

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("🔍 Filtres")
    selected_duration = st.radio(
        "⏳ Durée", 
        ["Indifférent", "Court (< 1h40)", "Moyen (1h40 - 2h20)", "Long (> 2h20)"]
    )
    st.markdown("---")
    selected_actors = st.multiselect("🎭 Acteurs / Réalisateurs", sorted_actors, placeholder="Ex: Leonardo DiCaprio")

# --- ZONE DE RECHERCHE ---
query = st.text_input("Que voulez-vous voir ?", placeholder="Ex: Un film de science-fiction psychologique...")

if (query != st.session_state.last_query or st.button("Rechercher 🚀")) and query:
    st.session_state.last_query = query
    st.session_state.num_visible = 5
    
    filters = {
        'duration': selected_duration,
        'actors': selected_actors
    }
    
    with st.spinner('Analyse IA ultra-rapide en cours...'):
        start_time = time.time()
        st.session_state.current_results = recommend_movies(query, filters, top_n=10)
        st.session_state.exec_time = time.time() - start_time

# --- AFFICHAGE DES RÉSULTATS ---
if st.session_state.current_results is not None:
    results = st.session_state.current_results
    visible_count = st.session_state.num_visible
    
    if len(results) == 0:
        st.warning("Aucun résultat ne correspond à tous vos filtres.")
    else:
        st.markdown(f"### 🎯 Top Recommandations ({min(visible_count, len(results))} affichés)")
        st.caption(f"⏱️ Recherche effectuée en **{st.session_state.exec_time:.2f} secondes**.")
        
        for i in range(0, visible_count, 5):
            cols = st.columns(5)
            for j in range(5):
                idx = i + j
                if idx < len(results) and idx < visible_count:
                    movie = results.iloc[idx]

                    with cols[j]:
                        poster_val = movie.get('poster_url')
                        poster = poster_val if pd.notna(poster_val) and bool(poster_val) else "https://via.placeholder.com/300x450?text=No+Image"
                        st.image(poster, use_container_width=True) 
                        
                        st.markdown(f"**{movie['title']}**")
                        
                        tagline = movie.get('tagline', '')
                        if pd.notna(tagline) and tagline:
                            st.markdown(f"*{tagline}*")
                        
                        year = str(movie['release_date'])[:4] if pd.notna(movie.get('release_date')) else "N/A"
                        st.caption(f"📅 {year} | ⭐ {movie.get('vote_average', 'N/A')}/10")
                        
                        genres = movie.get('genres', [])
                        if isinstance(genres, list) and genres:
                            st.caption(f"🏷️ {', '.join(genres[:3])}")

                        with st.expander("Infos"):
                            trailer_key = movie.get('trailer_key', '')
                            if pd.notna(trailer_key) and trailer_key:
                                st.video(f"https://www.youtube.com/watch?v={trailer_key}")
                                
                            actors = movie.get('actors', [])
                            director = movie.get('director', 'Inconnu')
                            actors_str = ", ".join(actors) if isinstance(actors, list) else str(actors)
                            st.write(f"**Réalisateur:** {director}")
                            st.write(f"**Casting:** {actors_str}")
                            
                            overview = str(movie.get('overview', 'Résumé indisponible'))
                            st.write(f"**Résumé:** {overview[:250]}...")
                            
        if st.session_state.num_visible < 25 and st.session_state.num_visible < len(results):
             if st.button("Afficher 5 films de plus ➕"):
                st.session_state.num_visible += 5
                st.rerun()