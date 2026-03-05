import pandas as pd
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

FILENAME = 'movies_data_ultimate.pkl'
FILENAME_CLEAN = 'movies_data_ultimate_clean.pkl'

# 1. CHARGEMENT COMPLET (Data + Matrices)
print(f"📂 Chargement du fichier '{FILENAME}'...")
try:
    with open(FILENAME, 'rb') as f:
        data = pickle.load(f)
    df = data['df']
    emb_overview = data['emb_overview']
    emb_metadata = data['emb_metadata']
    emb_reviews = data['emb_reviews']
    emb_cast = data['emb_cast']
except FileNotFoundError:
    print("❌ Fichier introuvable.")
    exit()

print(f"✅ Base chargée : {len(df)} films.")
print("-" * 30)

# 2. AUDIT ET ÉTAT DES LIEUX (STATISTIQUES & VISUALISATION)
# On remplit les vides pour analyser le texte
metadata_series = df['txt_metadata'].fillna("")
metadata_lengths = metadata_series.str.len()

print("\n📊 --- ÉTAT DES LIEUX AVANT NETTOYAGE ---")

# Calcul des stats clés
mean_len = metadata_lengths.mean()
median_len = metadata_lengths.median()
count_under_25 = (metadata_lengths < 25).sum()
percent_under_25 = (count_under_25 / len(df)) * 100

print(f"📏 Longueur moyenne des métadonnées : {mean_len:.1f} caractères")
print(f"📏 Longueur médiane : {median_len:.0f} caractères")
print(f"📉 Films très pauvres (< 25 chars) : {count_under_25} films (soit {percent_under_25:.2f}%)")

# Visualisation graphique
print("🎨 Génération des graphiques de distribution...")

plt.figure(figsize=(14, 6))

# Graphique 1 : Vue d'ensemble
plt.subplot(1, 2, 1)
sns.histplot(metadata_lengths, bins=50, kde=True, color='skyblue')
plt.axvline(x=25, color='red', linestyle='--', label='Seuil de suppression (25)')
plt.title('Distribution globale de la richesse des métadonnées')
plt.xlabel('Nombre de caractères (Genres + Mots-clés + Tagline)')
plt.ylabel('Nombre de films')
plt.legend()

# Graphique 2 : Zoom sur la zone critique
plt.subplot(1, 2, 2)
# On zoome sur les films qui ont moins de 150 caractères pour voir les "mauvais"
sns.histplot(metadata_lengths[metadata_lengths < 150], bins=30, color='orange')
plt.axvline(x=25, color='red', linestyle='--', linewidth=2, label='Zone de danger (<25)')
plt.title('Zoom sur les films pauvres en données')
plt.xlabel('Nombre de caractères')
plt.legend()

plt.tight_layout()
plt.show()

print("-" * 30)

# 3. DÉFINITION DES CRITÈRES DE SUPPRESSION
# Critère A : Longueur très faible (< 25 caractères)
mask_short = metadata_series.str.len() < 25

# Critère B : Structure vide "Genre. ." (Pas de keywords ni tagline)
# Regex qui cherche une chaine finissant par un point, des espaces éventuels, et un point final.
mask_empty = metadata_series.str.contains(r'\.\s*\.$', regex=True)

# Union des critères
indices_to_drop = df[mask_short | mask_empty].index

print(f"🔍 ANALYSE FINE : {len(indices_to_drop)} films identifiés comme 'à supprimer'.")

if len(indices_to_drop) > 0:
    print("Exemples de films ciblés :")
    print(df.loc[indices_to_drop[:5]][['title', 'txt_metadata']])
else:
    print("Aucun film ne correspond aux critères de suppression.")

# 4. NETTOYAGE (Suppression synchronisée)
keep_mask = np.ones(len(df), dtype=bool)

# Gestion des index non-alignés (sécurité)
if not df.index.equals(pd.RangeIndex(len(df))):
    print("⚠️ Réindexation du DataFrame nécessaire...")
    df = df.reset_index(drop=True)
    # Recalcul des masques sur le nouvel index pour être sûr
    metadata_series = df['txt_metadata'].fillna("")
    mask_short = metadata_series.str.len() < 25
    mask_empty = metadata_series.str.contains(r'\.\s*\.$', regex=True)
    indices_to_drop = df[mask_short | mask_empty].index

keep_mask[indices_to_drop] = False

# Filtrage du DataFrame
df_clean = df[keep_mask].reset_index(drop=True)

# Filtrage des 4 Matrices NumPy
emb_overview_clean = emb_overview[keep_mask]
emb_metadata_clean = emb_metadata[keep_mask]
emb_reviews_clean  = emb_reviews[keep_mask]
emb_cast_clean     = emb_cast[keep_mask]

print(f"📉 Taille avant : {len(df)}")
print(f"📈 Taille après : {len(df_clean)}")
print(f"🗑️ Total supprimé : {len(df) - len(df_clean)}")

# 5. SAUVEGARDE
print(f"💾 Sauvegarde dans '{FILENAME_CLEAN}'...")
with open(FILENAME_CLEAN, 'wb') as f:
    pickle.dump({
        'df': df_clean,
        'emb_overview': emb_overview_clean,
        'emb_metadata': emb_metadata_clean,
        'emb_reviews': emb_reviews_clean,
        'emb_cast': emb_cast_clean
    }, f)

print("🎉 Nettoyage terminé ! Le fichier clean est prêt.")