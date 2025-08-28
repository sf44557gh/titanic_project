# =====================================
# src/reporting.py
# =====================================
import pandas as pd

_METRIC_COLS = [
    "CV Accuracy", "Train Accuracy", "Gap (CV - Train)",
    "F1 Score", "Recall", "AUC", "Weighted Metric"
]

_BASE_COLS = [
    "Run_ID", "Timestamp", "Model", "Use_IsAlone", "Use_Cabin", "Best_Params"
] + _METRIC_COLS

def _short_params(params: dict, max_len: int = 80) -> str:
    s = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in sorted(params.items()))
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

def make_comparison_table(df_results: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
    df = df_results.copy()
    cols = [c for c in _BASE_COLS if c in df.columns]
    df = df[cols]

    df["Overfit?"] = (df["Gap (CV - Train)"] < -0.02)
    df["Underfit?"] = (df["Train Accuracy"] < 0.80)

    if "Best_Params" in df.columns:
        df["Best_Params_short"] = df["Best_Params"].apply(
            lambda p: _short_params(p) if isinstance(p, dict) else str(p)
        )

    df = df.sort_values(
        by=["Weighted Metric", "CV Accuracy", "F1 Score"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    if top_k is not None:
        df = df.head(top_k)

    ordered = [
        "Run_ID", "Timestamp", "Model", "Use_IsAlone", "Use_Cabin",
        "Best_Params_short",
        "Weighted Metric", "CV Accuracy", "F1 Score", "Recall", "AUC",
        "Train Accuracy", "Gap (CV - Train)", "Overfit?", "Underfit?"
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
    }
    return (
        df.style
        .format(fmt)
        .highlight_max(subset=["Weighted Metric", "CV Accuracy", "F1 Score", "Recall", "AUC"], color="#d4f4dd")
        .highlight_min(subset=["Gap (CV - Train)"], color="#fde2e1")
        .set_properties(subset=["Best_Params_short"], **{"white-space": "nowrap"})
    )
