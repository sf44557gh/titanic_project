'''
# src/training.py
import os
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)

def train_models(df_train, df_test, results_file="results/results_live.csv"):
    from src.preprocessing import clean_df, get_features, make_preprocessor
    from src.models import get_models_and_params, weighted_metric

    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    df_train = clean_df(df_train)
    df_test = clean_df(df_test)
    results_list = []
    run_id_start = 1
    run_id = run_id_start

    X_full, y_full = df_train.drop("Survived", axis=1), df_train["Survived"]

    for use_isalone in [True, False]:
        for use_cabin in [False, True]:
            features = get_features(df_train, use_isalone, use_cabin)
            X, y = df_train[features], y_full
            preprocessor = make_preprocessor(df_train, features)
            models = get_models_and_params()

            for model_name, (model, param_grid) in models.items():
                pipe = Pipeline([("preprocessor", preprocessor), ("clf", model)])
                grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
                grid.fit(X, y)

                y_pred_train = grid.predict(X)
                train_acc = accuracy_score(y, y_pred_train)
                cv_acc = grid.best_score_
                f1 = f1_score(y, y_pred_train)
                recall = recall_score(y, y_pred_train)
                auc = roc_auc_score(y, grid.predict_proba(X)[:,1]) if hasattr(grid.best_estimator_["clf"], "predict_proba") else float('nan')
                wm = weighted_metric(cv_acc, f1, recall, auc)
                gap = cv_acc - train_acc

                res = {
                    "Run_ID": run_id,
                    "Model": model_name,
                    "Use_IsAlone": use_isalone,
                    "Use_Cabin": use_cabin,
                    "Best_Params": grid.best_params_,
                    "Train Accuracy": train_acc,
                    "CV Accuracy": cv_acc,
                    "Gap (CV - Train)": gap,
                    "F1 Score": f1,
                    "Recall": recall,
                    "AUC": auc,
                    "Weighted Metric": wm,
                    "best_pipe": grid.best_estimator_
                }
                results_list.append(res)

                # Append dans CSV live
                df_res = pd.DataFrame([res])
                mode = "a" if os.path.exists(results_file) else "w"
                header = not os.path.exists(results_file)
                df_res.drop(columns=["best_pipe"]).to_csv(results_file, mode=mode, header=header, index=False)

                logger.info("Résultat Run_ID=%s enregistré", run_id)
                run_id += 1

    return pd.DataFrame(results_list)

'''
# src/training.py
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

logger = logging.getLogger(__name__)

def _next_run_id(results_file: Path) -> int:
    if results_file.exists() and results_file.stat().st_size > 0:
        try:
            prev = pd.read_csv(results_file, usecols=["Run_ID"])
            if not prev.empty:
                return int(pd.to_numeric(prev["Run_ID"], errors="coerce").fillna(0).astype(int).max()) + 1
        except Exception as e:
            logger.warning("Lecture Run_ID impossible (%s). Repart à 1.", e)
    return 1

def train_models(df_train, results_file: str | Path | None = None):
    from src.preprocessing import clean_df, get_features, make_preprocessor
    from src.models import get_models_and_params, weighted_metric

    # Racine du projet = un cran au-dessus de src/
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = Path(results_file) if results_file else (results_dir / "results_live.csv")

    # Run_ID persistant
    run_id = _next_run_id(results_file)

    # Entraînement
    df_train = clean_df(df_train)
    results_list = []

    X_full, y_full = df_train.drop("Survived", axis=1), df_train["Survived"]

    for use_isalone in [True, False]:
        for use_cabin in [False, True]:
            features = get_features(df_train, use_isalone, use_cabin)
            X, y = df_train[features], y_full

            preprocessor = make_preprocessor(df_train, features)
            models = get_models_and_params()

            for model_name, (model, param_grid) in models.items():
                pipe = Pipeline([("preprocessor", preprocessor), ("clf", model)])
                grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
                grid.fit(X, y)

                y_pred_train = grid.predict(X)
                train_acc = accuracy_score(y, y_pred_train)
                cv_acc = grid.best_score_
                f1 = f1_score(y, y_pred_train)
                recall = recall_score(y, y_pred_train)
                auc = roc_auc_score(y, grid.predict_proba(X)[:, 1]) if hasattr(grid.best_estimator_["clf"], "predict_proba") else np.nan
                wm = weighted_metric(cv_acc, f1, recall, auc)
                gap = cv_acc - train_acc

                res = {
                    "Run_ID": run_id,
                    "Model": model_name,
                    "Use_IsAlone": use_isalone,
                    "Use_Cabin": use_cabin,
                    "Best_Params": grid.best_params_,
                    "Train Accuracy": train_acc,
                    "CV Accuracy": cv_acc,
                    "Gap (CV - Train)": gap,
                    "F1 Score": f1,
                    "Recall": recall,
                    "AUC": auc,
                    "Weighted Metric": wm,
                    "best_pipe": grid.best_estimator_
                }
                results_list.append(res)

                # Append unique dans C:\Users\cdiac\Desktop\KaggleProjects\titanic_project\results\results_live.csv
                df_res = pd.DataFrame([res])
                mode = "a" if results_file.exists() else "w"
                header = not results_file.exists() or results_file.stat().st_size == 0
                df_res.drop(columns=["best_pipe"]).to_csv(results_file, mode=mode, header=header, index=False)

                logger.info("Résultat enregistré dans %s (Run_ID=%s)", results_file, run_id)
                run_id += 1

    return pd.DataFrame(results_list)
