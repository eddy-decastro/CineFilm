# 🎬 Projet Film : Analyse et Recommandation par IA

Bienvenue dans le dépôt du **Projet Film**. Cette application exploite l'Intelligence Artificielle et le Web Scraping pour recommander des films

L'architecture est construite sur deux piliers : un **Backend puissant** gérant la logique IA/Data, et un **Frontend interactif** pour l'utilisateur final.

---

## 🛠️ Technologies Utilisées

Ce projet repose sur une stack Data/Développement moderne :

* **Frontend (Interface Utilisateur) :** [Streamlit](https://streamlit.io/)
* **Backend (API) :** [FastAPI](https://fastapi.tiangolo.com/) (serveur ASGI via Uvicorn)
* **Machine Learning & Deep Learning :** PyTorch, Scikit-Learn
* **Data & Scraping :** Pandas, Numpy, BeautifulSoup4, Playwright
* **Visualisation :** Matplotlib, Seaborn

---

## ⚙️ Prérequis et Installation

Pour faire tourner ce projet sur votre machine, assurez-vous d'avoir installé Python (version 3.9 ou supérieure recommandée).

**1. Cloner le projet**
```bash
git clone [URL_DE_TON_DEPOT_GITHUB_SI_TU_EN_AS_UN]
cd Film

Film/
│
├── api.py                  # Code du serveur FastAPI (Endpoints, logique IA)
├── app.py                  # Code de l'interface graphique Streamlit
├── requirements.txt        # Liste exhaustive des dépendances Python
├── .venv/                  # Environnement virtuel (à ne pas push sur Git)
└── README.md               # Documentation du projet

## 🧠 Méthodologie et Apprentissages

Ce projet a été un véritable terrain d'expérimentation. J'ai appris la réalité de la création d'une application de bout en bout. Voici en toute transparence les étapes franchies et les compétences acquises :

### 1. Collecte et Préparation des Données
* **La réalité du Web Scraping :** J'ai utilisé `Playwright` et `BeautifulSoup` pour extraire mes données. J'ai appris qu'un site web change, que les données sont souvent incomplètes et qu'il faut concevoir des scripts robustes pour contourner les blocages. J'ai aussi utilisé une api sur TMDB.


### 2. Modélisation et Intelligence Artificielle
Choix des modèles : J'ai expérimenté plusieurs modèle pour l'embedding des différentes parties de texte des films. Le dernier modèle utilisé est le E5.
Ce que j'ai appris : Il existe plein de modèle différents possédant différentes caractéristiques.
Le modèle E5 est dans notre cas assez bon et pas trop lourd.

### 3. Architecture Logicielle (Le vrai défi)
* **Passer du "Notebook" à "l'Application" :** Le plus grand défi a été de sortir du confort des Jupyter Notebooks pour structurer un vrai projet logiciel.
* **L'approche Microservices (API + Frontend) :** J'ai choisi de séparer la logique IA (FastAPI) de l'interface utilisateur (Streamlit). Cela m'a appris comment faire communiquer deux programmes entre eux via des requêtes HTTP (requêtes GET/POST).
* **Résolution de problèmes IT :** J'ai dû affronter des contraintes de sécurité réelles, comme le blocage d'exécutables (Device Guard) sur des machines restreintes, que j'ai résolu en adaptant mes commandes d'exécution backend (`python -m uvicorn`).

### 🎯 Bilan Personnel
Si je devais refaire ce projet aujourd'hui, j'irai d'abord lire la littérature sur les méthodes utiles. Ce projet m'a prouvé que le métier de Data Scientist / Ingénieur IA ne s'arrête pas aux mathématiques, mais englobe aussi le génie logiciel.
C'était ma première approche à l'analyse de texte via l'informatique (TALN).