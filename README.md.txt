# Titanic Project 🚢

Prédiction de la survie des passagers du Titanic (Kaggle).

## 📂 Organisation du projet
- `data/` : données brutes (non versionnées)
- `notebooks/` : notebooks Jupyter (exploration, modélisation)
- `src/` : scripts Python (préprocessing, modèles, training)
- `results/` : résultats d’expériences (scores, CSV)
- `submissions/` : fichiers pour soumission Kaggle
- `models/` : modèles sauvegardés (`.pkl`)

## ⚡ Installation
Cloner le repo et installer les dépendances :
```bash
git clone https://github.com/tonpseudo/titanic_project.git
cd titanic_project
pip install -r requirements.txt


titanic_project/
│
├── data/ # Données locales (non versionnées)
├── notebooks/ # Jupyter notebooks (exploration & modélisation)
├── src/ # Code Python modulaire (préprocessing, modèles, etc.)
├── results/ # Résultats intermédiaires (non versionnés)
├── models/ # Modèles sauvegardés (non versionnés)
├── submissions/ # Fichiers de soumission Kaggle (non versionnés)
│
├── README.md # Ce fichier
├── requirements.txt # Dépendances Python
└── .gitignore # Fichiers/dossiers à ignorer