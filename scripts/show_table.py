# scripts/show_table.py
from pathlib import Path
import pandas as pd
from src.reporting import make_comparison_table

def main():
    project_root = Path(__file__).resolve().parents[1]
    results_csv = project_root / "results" / "results_live.csv"
    if not results_csv.exists():
        raise FileNotFoundError(f"Fichier introuvable: {results_csv}. Lance d'abord l'entraînement pour le créer.")

    df = pd.read_csv(results_csv)
    table = make_comparison_table(df, top_k=20)
    print(table.to_string(index=False))

if __name__ == "__main__":
    main()
