# =====================================
# src/reporting.py
# =====================================
import pandas as pd

# Colonnes métriques de base
_METRIC_COLS = [
    "CV Accuracy", "Train Accuracy", "Gap (CV - Train)",
    "F1 Score", "Recall", "AUC", "Weighted Metric"
]

# Colonnes d’instrumentation temps (facultatives)
_TIME_COLS = [
    "Grid_Size", "Search_Wall_Time_s",
    "Mean_Fit_Time_Best_s", "Mean_Score_Time_Best_s"
]

# Socle minimal + tags d’expérimentation
_BASE_COLS = [
    "Phase", "Experiment", "Run_ID", "Timestamp",
    "Model", "Use_IsAlone", "Use_Cabin", "Best_Params"
] + _METRIC_COLS + _TIME_COLS

def _short_params(params: dict, max_len: int = 80) -> str:
    s = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in sorted(params.items()))
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

def make_comparison_table(df_results: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
    df = df_results.copy()

    # Ne garder que les colonnes existantes parmi la liste cible
    cols = [c for c in _BASE_COLS if c in df.columns]
    df = df[cols]

    # Flags lecture
    if "Gap (CV - Train)" in df.columns:
        df["Overfit?"] = (df["Gap (CV - Train)"] < -0.02)
    if "Train Accuracy" in df.columns:
        df["Underfit?"] = (df["Train Accuracy"] < 0.80)

    # Raccourci paramètres
    if "Best_Params" in df.columns:
        df["Best_Params_short"] = df["Best_Params"].apply(
            lambda p: _short_params(p) if isinstance(p, dict) else str(p)
        )

    # Ordre Phase A puis B, puis tri par métriques
    if "Phase" in df.columns:
        phase_order = {"A": 0, "B": 1}
        df["_phase_order"] = df["Phase"].map(phase_order).fillna(99).astype(int)
        sort_keys = ["_phase_order", "Weighted Metric", "CV Accuracy", "F1 Score"]
        asc = [True, False, False, False]
    else:
        sort_keys = ["Weighted Metric", "CV Accuracy", "F1 Score"]
        asc = [False, False, False]

    df = df.sort_values(by=sort_keys, ascending=asc).reset_index(drop=True)
    if "_phase_order" in df.columns:
        df = df.drop(columns=["_phase_order"])

    if top_k is not None:
        df = df.head(top_k)

    ordered = [
        "Phase", "Experiment", "Run_ID", "Timestamp", "Model",
        "Use_IsAlone", "Use_Cabin", "Best_Params_short",
        "Weighted Metric", "CV Accuracy", "F1 Score", "Recall", "AUC",
        "Train Accuracy", "Gap (CV - Train)",
        "Grid_Size", "Search_Wall_Time_s", "Mean_Fit_Time_Best_s", "Mean_Score_Time_Best_s",
        "Overfit?", "Underfit?"
    ]
    df = df[[c for c in ordered if c in df.columns]]
    return df

def style_comparison_table(df: pd.DataFrame):
    fmt = {
        "Weighted Metric": "{:.3f}",
        "CV Accuracy": "{:.3f}",
        "F1 Score": "{:.3f}",
        "Recall": "{:.3f}",
        "AUC": "{:.3f}",
        "Train Accuracy": "{:.3f}",
        "Gap (CV - Train)": "{:+.3f}",
        "Search_Wall_Time_s": "{:.2f}",
        "Mean_Fit_Time_Best_s": "{:.3f}",
        "Mean_Score_Time_Best_s": "{:.3f}",
    }
    highlight_max_cols = [c for c in ["Weighted Metric", "CV Accuracy", "F1 Score", "Recall", "AUC"] if c in df.columns]
    return (
        df.style
        .format(fmt)
        .highlight_max(subset=highlight_max_cols, color="#d4f4dd")
        .highlight_min(subset=[c for c in ["Gap (CV - Train)"] if c in df.columns], color="#fde2e1")
        .set_properties(subset=["Best_Params_short"], **{"white-space": "nowrap"})
    )
