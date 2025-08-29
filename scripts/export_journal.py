# scripts/export_journal.py
# -*- coding: utf-8 -*-
"""
Génère/actualise results/journal_experimentation.xlsx à partir de results/results_live.csv
- Base métriques via src.reporting.make_comparison_table
- Ajoute colonnes projet (observations, prochaine action, etc.)
- Préserve les saisies manuelles si le journal existe déjà (merge par clés)
- Ajoute des lignes planifiées pour Stacking et Nested CV si absentes
"""

from pathlib import Path
import sys
import pandas as pd

# --- Réglages projet ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CSV_RESULTS = ROOT / "results" / "results_live.csv"
XLSX_JOURNAL = ROOT / "results" / "journal_experimentation.xlsx"

PROJECT_COLS = [
    "Notes / Observations",
    "Décision finale (Phase)",
    "Prochaine action",
    "Score Kaggle",
    "Version code / commit Git",
]

KEYS = ["Phase", "Experiment", "Run_ID", "Model"]


def _import_make_comparison_table():
    from src.reporting import make_comparison_table  # type: ignore
    return make_comparison_table


def _ensure_project_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in PROJECT_COLS:
        if c not in df.columns:
            df[c] = ""
    return df


def _with_project_columns(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    if "Run_ID" in out.columns and "Experiment" in out.columns:
        out["ID / Tag Run"] = out.apply(
            lambda r: f"{r['Experiment']}#R{int(r['Run_ID'])}" if pd.notna(r.get("Run_ID")) else "",
            axis=1,
        )
        cols = ["ID / Tag Run"] + [c for c in out.columns if c != "ID / Tag Run"]
        out = out[cols]

    if "Search_Wall_Time_s" in out.columns and "Durée calcul (Wall Time)" not in out.columns:
        out["Durée calcul (Wall Time)"] = out["Search_Wall_Time_s"]

    out = _ensure_project_cols(out)
    return out


def _merge_preserve_project_fields(new_table: pd.DataFrame, old_journal: pd.DataFrame) -> pd.DataFrame:
    old_journal = _ensure_project_cols(old_journal)
    keep_cols = [c for c in PROJECT_COLS if c in old_journal.columns]
    old_subset = old_journal[KEYS + keep_cols].drop_duplicates()

    merged = new_table.merge(old_subset, on=KEYS, how="left", suffixes=("", "_old"))

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


def main():
    if not CSV_RESULTS.exists():
        raise FileNotFoundError(f"{CSV_RESULTS} introuvable. Lance d’abord des runs pour créer results_live.csv.")

    make_comparison_table = _import_make_comparison_table()
    df_results = pd.read_csv(CSV_RESULTS)
    base_table = make_comparison_table(df_results, top_k=None)
    new_table = _with_project_columns(base_table)

    if XLSX_JOURNAL.exists():
        old = pd.read_excel(XLSX_JOURNAL)
        merged = _merge_preserve_project_fields(new_table, old)
    else:
        merged = new_table

    final = _add_planned_rows(merged)

    XLSX_JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    try:
        final.to_excel(XLSX_JOURNAL, index=False)
    except ModuleNotFoundError:
        sys.stderr.write("Erreur : openpyxl n’est pas installé. Installez-le avec:\n\n    pip install openpyxl\n\n")
        sys.exit(1)

    print(f"OK -> {XLSX_JOURNAL}")


if __name__ == "__main__":
    main()
