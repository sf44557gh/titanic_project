# scripts/export_journal.py
# -*- coding: utf-8 -*-
"""
Génère/actualise results/journal_experimentation.xlsx à partir de results/results_live.csv
- Normalisation via src.reporting.make_comparison_table
- Filtre "dernier run par (Experiment, Model)" pour éviter la pollution des tests à blanc
- Ajoute colonnes projet (observations, prochaine action, etc.)
- Préserve les saisies manuelles si le journal existe (merge par clés)
- Ajoute des lignes planifiées pour Stacking et Nested CV si absentes
- Message clair si openpyxl n'est pas installé
"""

from pathlib import Path
import sys
import pandas as pd

# --- Réglages ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CSV_RESULTS = ROOT / "results" / "results_live.csv"
XLSX_JOURNAL = ROOT / "results" / "journal_experimentation.xlsx"

# Active le filtre "dernier run par (Experiment, Model)"
KEEP_LAST_PER_EXPERIMENT_MODEL = True

PROJECT_COLS = [
    "Notes / Observations",
    "Décision finale (Phase)",
    "Prochaine action",
    "Score Kaggle",
    "Version code / commit Git",
]

# Clés d’alignement pour préserver les colonnes projet
KEYS = ["Phase", "Experiment", "Run_ID", "Model"]


# --- Imports paresseux ------------------------------------------------------
def _import_make_comparison_table():
    from src.reporting import make_comparison_table  # type: ignore
    return make_comparison_table


# --- Utils ------------------------------------------------------------------
def _ensure_project_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in PROJECT_COLS:
        if c not in df.columns:
            df[c] = ""
    return df


def _with_project_columns(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()

    # ID / Tag Run
    if "Run_ID" in out.columns and "Experiment" in out.columns:
        out["ID / Tag Run"] = out.apply(
            lambda r: f"{r['Experiment']}#R{int(r['Run_ID'])}" if pd.notna(r.get("Run_ID")) else "",
            axis=1,
        )
        out = out[["ID / Tag Run"] + [c for c in out.columns if c != "ID / Tag Run"]]

    # Durée calcul (Wall Time)
    if "Search_Wall_Time_s" in out.columns and "Durée calcul (Wall Time)" not in out.columns:
        out["Durée calcul (Wall Time)"] = out["Search_Wall_Time_s"]

    # Colonnes projet vides
    out = _ensure_project_cols(out)
    return out


def _merge_preserve_project_fields(new_table: pd.DataFrame, old_journal: pd.DataFrame) -> pd.DataFrame:
    old_journal = _ensure_project_cols(old_journal)
    keep_cols = [c for c in PROJECT_COLS if c in old_journal.columns]
    old_subset = old_journal[KEYS + keep_cols].drop_duplicates()

    merged = new_table.merge(old_subset, on=KEYS, how="left", suffixes=("", "_old"))

    # Priorité aux anciennes saisies si présentes
    for c in PROJECT_COLS:
        if c + "_old" in merged.columns:
            merged[c] = merged[c + "_old"].where(merged[c + "_old"].notna(), merged[c])
            merged.drop(columns=[c + "_old"], inplace=True)

    merged = _ensure_project_cols(merged)
    return merged


def _add_planned_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = list(out.columns)

    def exists(expr):
        mask = pd.Series([True] * len(out))
        for col, val in expr.items():
            mask &= out[col].astype(str).str.contains(val) if col == "Experiment" else (out[col] == val)
        return bool(mask.any())

    def empty_row():
        return {c: "" for c in cols}

    planned = []

    if not exists({"Phase": "A", "Experiment": "Stack"}):
        r = empty_row()
        r.update({
            "Phase": "A",
            "Experiment": "A-STACK_v1",
            "Run_ID": -1,
            "Model": "Stacking(LogReg, RF, HGB)",
            "Prochaine action": "Lancer Stacking (objectif: +0.01 Weighted)",
        })
        planned.append(r)

    if not exists({"Phase": "A", "Experiment": "Nested"}):
        r = empty_row()
        r.update({
            "Phase": "A",
            "Experiment": "A-NESTED_v1",
            "Run_ID": -1,
            "Model": "Nested CV (outer=5, inner=3)",
            "Prochaine action": "Lancer Nested CV (stabilité ≤0.03)",
        })
        planned.append(r)

    if planned:
        out = pd.concat([out, pd.DataFrame(planned)], ignore_index=True)

    return out


def _keep_last_per_experiment_model(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Ne conserve que le dernier run par (Experiment, Model).
    Ordre de priorité pour définir "dernier":
      1) Timestamp parsé le plus récent si disponible
      2) Sinon Run_ID le plus grand
    """
    if not KEEP_LAST_PER_EXPERIMENT_MODEL:
        return df_results

    df = df_results.copy()

    # Parse Timestamp si présent
    if "Timestamp" in df.columns:
        df["_ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    else:
        df["_ts"] = pd.NaT

    # Marqueur d'ordre
    df["_order_key"] = df["_ts"]
    # Si Timestamp manquant, fallback sur Run_ID
    if "_order_key" in df and df["_order_key"].isna().any():
        if "Run_ID" in df.columns:
            # Remplit NaT par -inf, puis combine avec Run_ID pour tri stable
            # On crée une clé numérique: ts_epoch pour tri, sinon Run_ID
            ts_numeric = df["_order_key"].view("int64")
            ts_numeric = ts_numeric.where(df["_order_key"].notna(), -1)  # -1 pour NaT
            df["_order_num"] = ts_numeric
            df["_order_num"] = df["_order_num"].astype("int64")
            # Combine en priorisant ts, puis Run_ID
            df["_final_order"] = df[["_order_num", "Run_ID"]].apply(
                lambda r: (r["_order_num"], int(r["Run_ID"]) if pd.notna(r["Run_ID"]) else -1), axis=1
            )
        else:
            # Pas de Run_ID: on se contente de Timestamp
            df["_final_order"] = df["_order_key"]
    else:
        df["_final_order"] = df["_order_key"]

    # Groupby et sélection du dernier (max de la clé d'ordre)
    def pick_last(g):
        idx = g["_final_order"].idxmax()
        return g.loc[idx:idx]

    kept = (
        df.groupby(["Experiment", "Model"], group_keys=False)
          .apply(pick_last)
          .drop(columns=["_ts", "_order_key", "_order_num", "_final_order"], errors="ignore")
          .reset_index(drop=True)
    )
    return kept


# --- Main --------------------------------------------------------------------
def main():
    if not CSV_RESULTS.exists():
        raise FileNotFoundError(f"{CSV_RESULTS} introuvable. Lance d’abord des runs pour créer results_live.csv.")

    make_comparison_table = _import_make_comparison_table()

    # 1) Lecture des résultats bruts
    df_results = pd.read_csv(CSV_RESULTS)

    # 2) Filtre: ne garder que le dernier run par (Experiment, Model)
    df_results = _keep_last_per_experiment_model(df_results)

    # 3) Table métriques normalisée
    base_table = make_comparison_table(df_results, top_k=None)

    # 4) Ajout colonnes projet, ID/Tag, Wall Time
    new_table = _with_project_columns(base_table)

    # 5) Préserver les saisies si journal existant
    if XLSX_JOURNAL.exists():
        old = pd.read_excel(XLSX_JOURNAL)
        merged = _merge_preserve_project_fields(new_table, old)
    else:
        merged = new_table

    # 6) Lignes planifiées
    final = _add_planned_rows(merged)

    # 7) Export
    XLSX_JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    try:
        final.to_excel(XLSX_JOURNAL, index=False)
    except ModuleNotFoundError:
        sys.stderr.write("Erreur : openpyxl n’est pas installé. Installez-le avec:\n\n    pip install openpyxl\n\n")
        sys.exit(1)

    print(f"OK -> {XLSX_JOURNAL}")


if __name__ == "__main__":
    main()
