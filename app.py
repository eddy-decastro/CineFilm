import streamlit as st
import requests

# --- CONFIGURATION ---
st.set_page_config(page_title="CinéFilm IA", layout="wide", page_icon="🍿")

API_URL = "http://127.0.0.1:8000/recommend"
#essaye de github
# --- GESTION DES THÈMES (CSS INJECTÉ) ---
def apply_theme(theme_name):
    if theme_name == "🍿 Style Netflix (Immersif)":
        css = """
        <style>
            /* Cacher les menus techniques */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            /* Effet sur les affiches */
            img {border-radius: 12px; transition: transform 0.3s ease; box-shadow: 0 10px 20px rgba(0,0,0,0.5);}
            img:hover {transform: scale(1.06); z-index: 10;}
            /* Bouton principal rouge */
            div.stButton > button {background-color: #E50914; color: white; border: none; border-radius: 6px; font-weight: bold;}
            div.stButton > button:hover {background-color: #B20710; color: white;}
        </style>
        """
    elif theme_name == "📝 Style Letterboxd (Cinéphile)":
        css = """
        <style>
            #MainMenu {visibility: hidden;}
            /* Affiches style cartes strictes */
            img {border-radius: 4px; border: 1px solid #445566;}
            img:hover {border: 1px solid #00E054;}
            /* Bouton principal vert */
            div.stButton > button {background-color: #2C3440; color: #00E054; border: 1px solid #00E054; border-radius: 4px;}
            div.stButton > button:hover {background-color: #00E054; color: #14181C;}
        </style>
        """
    else: # 🤖 Style Tech / Dashboard IA
        css = """
        <style>
            #MainMenu {visibility: hidden;}
            /* Affiches avec lueur tech */
            img {border-radius: 0px; border-bottom: 3px solid #00D2FF; filter: grayscale(20%); transition: filter 0.3s;}
            img:hover {filter: grayscale(0%); box-shadow: 0 0 15px rgba(0, 210, 255, 0.5);}
            /* Bouton principal bleu cyber */
            div.stButton > button {background-color: transparent; color: #00D2FF; border: 2px solid #00D2FF; text-transform: uppercase; letter-spacing: 1px;}
            div.stButton > button:hover {background-color: #00D2FF; color: black; box-shadow: 0 0 10px #00D2FF;}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

# --- VARIABLES DE SESSION ---
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'exec_time' not in st.session_state:
    st.session_state.exec_time = 0.0

# --- BARRE LATÉRALE ---
with st.sidebar:
    st.header("🎨 Personnalisation")
    selected_theme = st.selectbox(
        "Choisissez votre interface :",
        ["🍿 Style Netflix (Immersif)", "📝 Style Letterboxd (Cinéphile)", "🤖 Style Tech / Dashboard IA"]
    )
    apply_theme(selected_theme) # On applique le CSS choisi instantanément
    
    st.markdown("---")
    st.header("🔍 Filtres de Recherche")
    selected_duration = st.select_slider(
        "⏳ Durée du film",
        options=["Indifférent", "Court (< 1h40)", "Moyen (1h40 - 2h20)", "Long (> 2h20)"]
    )
    actors_input = st.text_input("🎭 Casting requis", placeholder="Ex: Di Caprio, Brad Pitt")
    selected_actors = [a.strip() for a in actors_input.split(",")] if actors_input else []

# --- EN-TÊTE PRINCIPAL ---
st.title("🎬 CinéFilm IA")
st.markdown("Décrivez votre envie, l'algorithme trouve le film parfait en un clin d'œil.")

# --- ZONE DE RECHERCHE ---
query = st.text_input("", placeholder="Ex: Un thriller psychologique se déroulant dans un sous-marin...", label_visibility="collapsed")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_clicked = st.button("🚀 Lancer la recherche IA", use_container_width=True)

# --- LOGIQUE DE RECHERCHE ---
if (query != st.session_state.last_query or search_clicked) and query:
    st.session_state.last_query = query
    
    payload = {
        "query": query,
        "duration": selected_duration,
        "actors": selected_actors,
        "top_n": 10
    }
    
    with st.spinner('Connexion à l\'API IA...'):
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            st.session_state.current_results = data.get("results", [])
            st.session_state.exec_time = data.get("time_sec", 0.0)
            
        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de joindre l'API. Vérifiez que le serveur FastAPI tourne sur le port 8000.")
            st.session_state.current_results = None
        except Exception as e:
            st.error(f"❌ Erreur inattendue : {e}")
            st.session_state.current_results = None

# --- AFFICHAGE DES RÉSULTATS OU DE L'ACCUEIL ---
if not query and st.session_state.current_results is None:
    # L'ÉCRAN D'ACCUEIL (Empty State)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("💡 **Bienvenue sur CinéFilm IA !** Ne cherchez pas par titre, cherchez par *concept*. Essayez par exemple :")
    c1, c2, c3 = st.columns(3)
    c1.success("« Une comédie absurde sur le sens de la vie et la mort »")
    c2.warning("« Un huis-clos angoissant dans l'espace sans aucun bruit »")
    c3.error("« Un braquage de banque ultra réaliste avec une trahison »")

elif st.session_state.current_results is not None:
    results = st.session_state.current_results
    
    if len(results) == 0:
        st.warning("Aucun résultat ne correspond à vos filtres stricts. Essayez d'élargir la recherche.")
    else:
        st.markdown(f"### 🎯 Top {len(results)} Recommandations")
        st.caption(f"⚡ Analysé parmi 6000 films en **{st.session_state.exec_time} secondes** par le modèle E5-Large.")
        st.markdown("---")
        
        # --- LA GRILLE DE FILMS (La partie qui manquait) ---
        
        for i in range(0, len(results), 5):
            cols = st.columns(5)
            for j in range(5):
                idx = i + j
                if idx < len(results):
                    movie = results[idx]

                    with cols[j]:
                        # 1. AFFICHE
                        poster_val = movie.get('poster_url')
                        poster = poster_val if poster_val else "https://via.placeholder.com/300x450?text=No+Image"
                        st.image(poster, use_container_width=True) 
                        
                        # 2. TITRE & TAGLINE
                        st.markdown(f"**{movie.get('title', 'Titre inconnu')}**")
                        tagline = movie.get('tagline', '')
                        if tagline:
                            st.markdown(f"*{tagline}*")
                        
                        # 3. DATE & NOTE
                        year = str(movie.get('release_date', 'N/A'))[:4]
                        vote = movie.get('vote_average', 0)
                        st.caption(f"📅 {year} | ⭐ {vote}/10")
                        
                        # 4. JAUGE IA (Nouveauté UX)
                        ai_score = movie.get('ai_score', 0)
                        # On limite le score à 1.0 (100%) max pour éviter que la barre d'UI ne plante
                        safe_score = min(max(float(ai_score), 0.0), 1.0) 
                        match_pct = int(safe_score * 100)
                        st.progress(safe_score, text=f"🔥 Match IA : {match_pct}%")
                        
                        # 5. GENRES
                        genres = movie.get('genres', [])
                        if isinstance(genres, list) and genres:
                            st.caption(f"🏷️ {', '.join(genres[:3])}")

                        # 6. INFOS CACHÉES
                        with st.expander("Infos & Bande-Annonce"):
                            trailer_key = movie.get('trailer_key', '')
                            if trailer_key:
                                st.video(f"https://www.youtube.com/watch?v={trailer_key}")
                                
                            actors = movie.get('actors', [])
                            director = movie.get('director', 'Inconnu')
                            actors_str = ", ".join(actors) if isinstance(actors, list) else str(actors)
                            st.write(f"**Réalisateur:** {director}")
                            st.write(f"**Casting:** {actors_str}")
                            
                            overview = str(movie.get('overview', 'Résumé indisponible'))
                            st.write(f"**Résumé:** {overview[:250]}...")