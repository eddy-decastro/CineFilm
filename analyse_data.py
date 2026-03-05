import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration esthétique des graphiques
sns.set_theme(style="whitegrid")

def load_data(filepath='Data_movie.pkl'):
    """Charge uniquement le DataFrame depuis le fichier Pickle."""
    try:
        data = pd.read_pickle(filepath)
        return data['df']
    except Exception as e:
        print(f"❌ Erreur lors du chargement : {e}")
        return None

def plot_movies_per_year(df):
    """Génère une courbe d'évolution du nombre de films par année."""
    plt.figure(figsize=(12, 6))
    
    # Conversion propre des dates pour extraire l'année
    years = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    counts = years.value_counts().sort_index()
    
    # On limite l'affichage de 1980 à nos jours pour plus de lisibilité
    counts = counts[counts.index >= 1980]
    
    sns.lineplot(x=counts.index, y=counts.values, marker='o', color='#E50914', linewidth=2.5)
    plt.title('Évolution du nombre de films par année de sortie (depuis 1980)', fontsize=16, fontweight='bold')
    plt.xlabel('Année', fontsize=12)
    plt.ylabel('Nombre de films', fontsize=12)
    plt.tight_layout()
    plt.savefig('1_evolution_annees.png')
    plt.show()

def plot_text_lengths(df):
    """Analyse la distribution de la taille des textes pour l'IA."""
    plt.figure(figsize=(14, 6))
    
    # Calcul des longueurs en nombre de caractères
    df['overview_len'] = df['overview'].fillna('').astype(str).str.len()
    df['metadata_len'] = df['txt_metadata'].fillna('').astype(str).str.len()
    
    # Graphique 1 : Les résumés (Overview)
    plt.subplot(1, 2, 1)
    sns.histplot(df['overview_len'], bins=50, kde=True, color='purple')
    plt.title('Distribution de la taille des Résumés', fontsize=14, fontweight='bold')
    plt.xlabel('Nombre de caractères')
    plt.ylabel('Nombre de films')
    plt.axvline(df['overview_len'].median(), color='black', linestyle='--', label=f"Médiane: {df['overview_len'].median():.0f} car.")
    plt.legend()
    
    # Graphique 2 : Les métadonnées
    plt.subplot(1, 2, 2)
    sns.histplot(df['metadata_len'], bins=50, kde=True, color='orange')
    plt.title('Distribution de la taille des Métadonnées', fontsize=14, fontweight='bold')
    plt.xlabel('Nombre de caractères')
    plt.ylabel('Nombre de films')
    plt.axvline(df['metadata_len'].median(), color='black', linestyle='--', label=f"Médiane: {df['metadata_len'].median():.0f} car.")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('2_taille_textes.png')
    plt.show()

def plot_genre_ranking(df):
    """Crée un diagramme en barres du classement des genres."""
    plt.figure(figsize=(12, 8))
    
    all_genres = []
    for genres_list in df['genres'].dropna():
        if isinstance(genres_list, list):
            all_genres.extend(genres_list)
                
    genre_counts = pd.Series(all_genres).value_counts()
    
    sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='magma', hue=genre_counts.index, legend=False)
    plt.title('Classement des films par Genre', fontsize=16, fontweight='bold')
    plt.xlabel('Nombre d\'occurrences', fontsize=12)
    plt.ylabel('Genres', fontsize=12)
    plt.tight_layout()
    plt.savefig('3_classement_genres.png')
    plt.show()

def plot_average_rating_per_year(df):
    """Génère une courbe de l'évolution de la note moyenne par année."""
    plt.figure(figsize=(12, 6))
    
    # On crée une copie pour ne pas modifier le df original
    df_plot = df.copy()
    df_plot['year'] = pd.to_datetime(df_plot['release_date'], errors='coerce').dt.year
    
    # On groupe par année et on calcule la moyenne de la colonne 'vote_average'
    yearly_ratings = df_plot.groupby('year')['vote_average'].mean().dropna()
    
    # On filtre depuis 1980 pour correspondre au premier graphique
    yearly_ratings = yearly_ratings[yearly_ratings.index >= 1980]
    
    # Création de la courbe (couleur bleue pour la différencier du nombre de films)
    sns.lineplot(x=yearly_ratings.index, y=yearly_ratings.values, marker='s', color='#2E86C1', linewidth=2.5)
    
    plt.title('Évolution de la note moyenne des films par année (depuis 1980)', fontsize=16, fontweight='bold')
    plt.xlabel('Année', fontsize=12)
    plt.ylabel('Note Moyenne (/10)', fontsize=12)
    
    # On restreint l'axe Y entre 5 et 9 pour bien voir les variations (les notes TMDB sont rarement hors de cette plage)
    plt.ylim(5, 9) 
    
    plt.tight_layout()
    plt.savefig('4_moyenne_notes_annees.png')
    plt.show()

if __name__ == "__main__":
    print("⏳ Chargement des données (cela peut prendre quelques secondes)...")
    df = load_data()
    
    if df is not None:
        print(f"✅ Base de données chargée avec {len(df)} films.")
        
        print("\n📊 1/4 - Génération de la courbe des années...")
        plot_movies_per_year(df)
        
        print("📊 2/4 - Génération des histogrammes de taille de texte...")
        plot_text_lengths(df)
        
        print("📊 3/4 - Génération du classement des genres...")
        plot_genre_ranking(df)
        
        print("📊 4/4 - Génération de la courbe des notes moyennes...")
        plot_average_rating_per_year(df)
        
        print("\n🎉 Analyse terminée ! Les 4 images PNG ont été sauvegardées dans ton dossier.")
        