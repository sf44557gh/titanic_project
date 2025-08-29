# =====================================
# src/training.py — Phase A instrumentée (hyperparamètres)
# =====================================
# Ce module fournit une implémentation "propre" de la Phase A :
# - on gèle un set de features (ex.: use_isalone=True, use_cabin=False)
# - on optimise les hyperparamètres de chaque modèle (LogReg, RF, HGB)
# - on journalise chaque résultat dans results/results_live.csv
#   avec Phase="A", Experiment="...", et des colonnes de temps pour comparer les coûts.
#
# Tu réutiliseras ensuite le dict retourné {model_name: best_params}
# pour la Phase B (tests de features) dans un autre module.

from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime
from time import perf_counter
from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, ParameterGrid,
    cross_val_predict, cross_val_score
)
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Helpers I/O et gestion du fichier de résultats
# -------------------------------------------------------------------
def _project_root() -> Path:
    """
    Racine du projet = parent de src/.
    Permet d’avoir des chemins stables (ex.: results/).
    """
    return Path(__file__).resolve().parents[1]

def _results_path(results_file: str | Path | None) -> Path:
    """
    Retourne le chemin du CSV des résultats.
    Crée le dossier results/ si nécessaire.
    """
    results_dir = _project_root() / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return Path(results_file) if results_file else (results_dir / "results_live.csv")

def _next_run_id(results_file: Path) -> int:
    """
    Incrémente de façon robuste le Run_ID en lisant le CSV existant.
    """
    if results_file.exists() and results_file.stat().st_size > 0:
        try:
            prev = pd.read_csv(results_file, usecols=["Run_ID"])
            if not prev.empty:
                return int(pd.to_numeric(prev["Run_ID"], errors="coerce").fillna(0).astype(int).max()) + 1
        except Exception as e:
            logger.warning("Lecture Run_ID impossible (%s). Repart à 1.", e)
    return 1

def _save_result(results_file: Path, record: dict) -> None:
    """
    Append une ligne dans results_live.csv (sans objets non-sérialisables).
    """
    df_res = pd.DataFrame([record]).drop(columns=[c for c in ["best_pipe"] if c in record])
    mode = "a" if results_file.exists() else "w"
    header = not results_file.exists() or results_file.stat().st_size == 0
    df_res.to_csv(results_file, mode=mode, header=header, index=False)

# -------------------------------------------------------------------
# Helpers métriques
# -------------------------------------------------------------------
def _compute_cv_metrics(pipe: Pipeline, X, y, cv: StratifiedKFold) -> Tuple[float, float, float, float]:
    """
    Calcule des métriques comparables entre modèles via CV stratifiée.
    Retourne: (cv_accuracy, f1, recall, auc)
    - Accuracy via cross_val_score
    - F1/Recall via cross_val_predict (prédictions discrètes)
    - AUC via predict_proba (ou decision_function à défaut)
    """
    # Accuracy moyenne par CV
    cv_acc = float(np.mean(cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)))

    # Prédictions discrètes pour F1/Recall
    y_pred = cross_val_predict(pipe, X, y, cv=cv, method="predict", n_jobs=-1)
    f1 = f1_score(y, y_pred)
    recall = recall_score(y, y_pred)

    # AUC (probabilités si possible, sinon decision_function, sinon NaN)
    auc = np.nan
    try:
        y_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        auc = roc_auc_score(y, y_proba)
    except Exception:
        try:
            y_dec = cross_val_predict(pipe, X, y, cv=cv, method="decision_function", n_jobs=-1)
            auc = roc_auc_score(y, y_dec)
        except Exception:
            pass

    return cv_acc, f1, recall, auc

def _weighted_metric(cv_acc: float, f1: float, recall: float, auc: float,
                     w: Tuple[float, float, float, float] = (0.4, 0.2, 0.2, 0.2)) -> float:
    """
    Score composite (pondéré) pour classer les modèles.
    Par défaut: 40% Accuracy CV, 20% F1, 20% Recall, 20% AUC.
    """
    auc_safe = 0.0 if np.isnan(auc) else float(auc)
    return float(w[0]*cv_acc + w[1]*f1 + w[2]*recall + w[3]*auc_safe)

# -------------------------------------------------------------------
# Phase A : optimisation d’hyperparamètres (set de features gelé)
# -------------------------------------------------------------------
def train_hyperparams(df_train: pd.DataFrame,
                      use_isalone: bool = True,
                      use_cabin: bool = False,
                      results_file: str | Path | None = None,
                      experiment: str = "A-default",
                      param_grids_override: Optional[Dict[str, Tuple[Any, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Phase A : optimise les hyperparamètres des modèles sur un set de features gelé.
    Ecrit chaque modèle dans results/results_live.csv avec :
      - Phase="A", Experiment=<experiment>
      - métriques CV (Accuracy, F1, Recall, AUC), Train Accuracy, Gap
      - colonne synthèse Weighted Metric
      - colonnes de coût : Grid_Size, Search_Wall_Time_s, Mean_Fit_Time_Best_s, Mean_Score_Time_Best_s

    Paramètres
    ----------
    df_train : DataFrame d’entraînement (doit contenir 'Survived').
    use_isalone, use_cabin : flags de features (gels Phase A).
    results_file : chemin CSV de sortie (défaut: results/results_live.csv).
    experiment : tag libre pour tracer l’essai (ex.: "A-RF_HGB_LogReg_v2").
    param_grids_override : dictionnaire optionnel permettant de remplacer les
      modèles/grilles par défaut issus de src.models.get_models_and_params().
      Format: { "ModelName": (estimator, param_grid) }, où param_grid peut être
      un dict ou une liste de dicts (sklearn ParameterGrid compatible).

    Retour
    ------
    best_params_map : dict {model_name: best_params_dict} pour la Phase B.
    """
    # Import tardif pour éviter les dépendances circulaires au chargement du module
    from src.preprocessing import clean_df, get_features, make_preprocessor
    from src.models import get_models_and_params

    # 1) Sortie et Run_ID
    results_file = _results_path(results_file)
    run_id = _next_run_id(results_file)

    # 2) Nettoyage des données et split X/y
    df_train = clean_df(df_train)
    X_full, y_full = df_train.drop("Survived", axis=1), df_train["Survived"]

    # 3) Set de features gelé pour la Phase A
    features = get_features(df_train, use_isalone, use_cabin)
    X, y = df_train[features], y_full
    preprocessor = make_preprocessor(df_train, features)

    # 4) Modèles et grilles
    base_models = get_models_and_params()  # dict {name: (estimator, param_grid)}
    models = dict(base_models)
    if param_grids_override:
        # Remplace uniquement les entrées explicitement fournies
        for k, v in param_grids_override.items():
            models[k] = v

    best_params_map: Dict[str, Dict[str, Any]] = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 5) Boucle sur les modèles
    for model_name, (model, param_grid) in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("clf", model)])

        # 5.1 Taille de grille (dict ou liste de dicts)
        try:
            grid_size = len(list(ParameterGrid(param_grid)))
        except Exception:
            grid_size = None  # si param_grid n’est pas compatible (cas très rare)

        # 5.2 Chronométrage de la recherche d’hyperparamètres
        t0 = perf_counter()
        grid = GridSearchCV(
            pipe, param_grid,
            cv=cv, scoring="accuracy",
            n_jobs=-1, refit=False, return_train_score=False
        )
        grid.fit(X, y)
        search_wall_time = perf_counter() - t0

        # 5.3 Reconstitution du pipeline sur les best_params et calcul métriques CV comparables
        best_params_only = grid.best_params_ or {}
        best_pipe = Pipeline([("preprocessor", preprocessor), ("clf", model)])
        if best_params_only:
            best_pipe.set_params(**best_params_only)

        cv_acc, f1, recall, auc = _compute_cv_metrics(best_pipe, X, y, cv)

        # 5.4 Fit final sur tout X pour mesurer le gap (CV - Train)
        best_pipe.fit(X, y)
        y_pred_train = best_pipe.predict(X)
        train_acc = accuracy_score(y, y_pred_train)
        gap = float(cv_acc - train_acc)

        wm = _weighted_metric(cv_acc, f1, recall, auc)

        # 5.5 Temps moyens du meilleur candidat (issus de cv_results_)
        mean_fit_time_best = None
        mean_score_time_best = None
        try:
            idx = int(grid.best_index_)
            mean_fit_time_best = float(grid.cv_results_["mean_fit_time"][idx])
            mean_score_time_best = float(grid.cv_results_["mean_score_time"][idx])
        except Exception:
            pass  # champs resteront None si indisponibles

        # 5.6 Journalisation d’une ligne dans results_live.csv
        record = {
            "Phase": "A",
            "Experiment": experiment,
            "Run_ID": run_id,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Model": model_name,
            "Use_IsAlone": use_isalone,
            "Use_Cabin": use_cabin,
            "Best_Params": best_params_only,
            # Métriques
            "Train Accuracy": float(train_acc),
            "CV Accuracy": float(cv_acc),
            "Gap (CV - Train)": float(gap),
            "F1 Score": float(f1),
            "Recall": float(recall),
            "AUC": float(auc) if not np.isnan(auc) else np.nan,
            "Weighted Metric": float(wm),
            # Coûts/temps
            "Grid_Size": grid_size,
            "Search_Wall_Time_s": round(search_wall_time, 2),
            "Mean_Fit_Time_Best_s": mean_fit_time_best,
            "Mean_Score_Time_Best_s": mean_score_time_best,
            # Objet non sérialisé gardé en mémoire appelante si besoin
            "best_pipe": best_pipe,
        }
        _save_result(results_file, record)
        logger.info("[Phase A] %s - %s - Run_ID=%s - WallTime=%.2fs",
                    experiment, model_name, run_id, search_wall_time)

        # 5.7 Mémorise les meilleurs hyperparamètres du modèle
        best_params_map[model_name] = best_params_only
        run_id += 1

    return best_params_map
