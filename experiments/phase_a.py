# =====================================
# phase_a.py — Lancement Phase A
# =====================================
from pathlib import Path
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.training import train_hyperparams

# 1) Charger les données d'entraînement Titanic
df_train = pd.read_csv(Path("data/raw/train.csv"))

# 2) Définir des grilles enrichies pour chaque modèle
param_grids_override = {
    "RandomForest": (
        RandomForestClassifier(random_state=42),
        {
            "clf__n_estimators": [300, 500, 800],
            "clf__max_depth": [None, 6, 10],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4]
        }
    ),
    "HistGB": (
        HistGradientBoostingClassifier(random_state=42),
        {
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__max_leaf_nodes": [15, 31],
            "clf__max_depth": [None, 3, 5],
            "clf__min_samples_leaf": [10, 20]
        }
    ),
    "LogReg": (
        LogisticRegression(max_iter=2000, random_state=42),
        [
            {"clf__penalty":["l2"],"clf__C":[0.01,0.1,1,10],"clf__solver":["lbfgs","saga"]},
            {"clf__penalty":["l1"],"clf__C":[0.01,0.1,1,10],"clf__solver":["liblinear","saga"]},
            {"clf__penalty":["elasticnet"],"clf__l1_ratio":[0.2,0.5,0.8],"clf__C":[0.1,1],"clf__solver":["saga"]},
        ]
    )
}

# 3) Lancer la Phase A
best_params_map = train_hyperparams(
    df_train,
    use_isalone=True,   # set de features gelé pour Phase A
    use_cabin=False,
    experiment="A-RF_HGB_LogReg_v1",
    param_grids_override=param_grids_override
)

# 4) Sauvegarder les meilleurs hyperparamètres trouvés
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "best_params_phaseA.json", "w", encoding="utf-8") as f:
    json.dump(best_params_map, f, ensure_ascii=False, indent=2)

print("Best params Phase A sauvegardés -> results/best_params_phaseA.json")
