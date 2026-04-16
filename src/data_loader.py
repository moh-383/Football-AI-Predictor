"""
data_loader.py — Football AI Predictor v3.0
============================================
Changements vs v2.x :
  - 10 saisons par défaut (1516 → 2324) au lieu de 5
  - Rapport de qualité des données par saison
  - Vérification de l'intégrité temporelle après concat
"""

from __future__ import annotations

import os

import pandas as pd
import requests


# ──────────────────────────────────────────────────────────────────────────────
# Téléchargement
# ──────────────────────────────────────────────────────────────────────────────

# 10 saisons : 2015-16 → 2023-24 (≈3 040 matchs)
DEFAULT_SEASONS = [
    "1516", "1617", "1718", "1819",
    "1920", "2021", "2122", "2223", "2324",
]


def download_ligue1_data(seasons: list[str] = DEFAULT_SEASONS) -> pd.DataFrame:
    """
    Télécharge les données Ligue 1 pour les saisons demandées.

    Args:
        seasons: Codes de saisons (ex: ["2122", "2223", "2324"])

    Returns:
        pd.DataFrame: Toutes les saisons concaténées (données brutes)
    """
    os.makedirs("data/raw", exist_ok=True)
    base_url = "https://www.football-data.co.uk/mmz4281/{season}/F1.csv"
    all_dfs  = []

    print(f"  Téléchargement de {len(seasons)} saisons…\n")

    for season in seasons:
        url      = base_url.format(season=season)
        filepath = f"data/raw/ligue1_{season}.csv"

        # Cache local : évite le re-téléchargement
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, encoding="latin1")
            df["season"] = season
            all_dfs.append(df)
            print(f"  📂 {season} (cache)     : {len(df)} matchs")
            continue

        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                with open(filepath, "wb") as f:
                    f.write(response.content)
                df = pd.read_csv(filepath, encoding="latin1")
                df["season"] = season
                all_dfs.append(df)
                print(f"  ✅ {season} téléchargé  : {len(df)} matchs")
            else:
                print(f"  ❌ {season} HTTP {response.status_code} — ignoré")
        except requests.RequestException as e:
            print(f"  ❌ {season} erreur réseau : {e}")

    if not all_dfs:
        raise RuntimeError("Aucune donnée téléchargée. Vérifiez votre connexion.")

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n  Total brut : {len(combined)} matchs sur {len(all_dfs)} saisons")
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# Nettoyage
# ──────────────────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie et standardise les données brutes.

    Garanties :
        - Tri chronologique strict (critique pour anti-leakage)
        - Encodage cible : 0=ext gagne, 1=nul, 2=dom gagne
        - Suppression des lignes sans résultat final
    """
    cols_wanted = [
        "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HS", "AS", "HST", "AST", "HC", "AC",
        "HF", "AF", "HY", "AY", "HR", "AR", "season",
    ]
    cols_available = [c for c in cols_wanted if c in df.columns]
    df = df[cols_available].copy()

    rename_map = {
        "Date":     "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG":     "home_goals",
        "FTAG":     "away_goals",
        "FTR":      "result",
        "HS":       "home_shots",
        "AS":       "away_shots",
        "HST":      "home_shots_target",
        "AST":      "away_shots_target",
        "HC":       "home_corners",
        "AC":       "away_corners",
        "HF":       "home_fouls",
        "AF":       "away_fouls",
        "HY":       "home_yellow",
        "AY":       "away_yellow",
        "HR":       "home_red",
        "AR":       "away_red",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Parsing de date robuste (formats DD/MM/YY et DD/MM/YYYY coexistent)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"])

    # Encodage cible
    df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0})
    df = df.dropna(subset=["home_goals", "away_goals", "result"])

    # CRUCIAL : tri chronologique strict
    df = df.sort_values("date").reset_index(drop=True)

    return df


def print_data_quality_report(df: pd.DataFrame) -> None:
    """Affiche un rapport de qualité par saison."""
    print("\n  Rapport qualité par saison :")
    print(f"  {'Saison':<8} {'Matchs':>7} {'NaN HST':>8} {'H%':>6} {'D%':>6} {'A%':>6}")
    print(f"  {'─'*46}")

    for season, grp in df.groupby("season"):
        n       = len(grp)
        nan_hst = grp["home_shots_target"].isna().sum() if "home_shots_target" in grp else n
        h_pct   = (grp["result"] == "H").mean()
        d_pct   = (grp["result"] == "D").mean()
        a_pct   = (grp["result"] == "A").mean()
        print(
            f"  {season:<8} {n:>7} {nan_hst:>8} {h_pct:>5.1%} {d_pct:>5.1%} {a_pct:>5.1%}"
        )

    total = len(df)
    print(f"  {'─'*46}")
    print(f"  {'TOTAL':<8} {total:>7}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Téléchargement données Ligue 1")
    parser.add_argument(
        "--seasons", nargs="+", default=DEFAULT_SEASONS,
        help="Codes de saisons (ex: 2122 2223 2324)"
    )
    args = parser.parse_args()

    print(f"\n=== Téléchargement Ligue 1 — {len(args.seasons)} saisons ===\n")

    df_raw = download_ligue1_data(seasons=args.seasons)

    print("\nNettoyage des données…")
    df_clean = clean_data(df_raw)

    print_data_quality_report(df_clean)

    os.makedirs("data/processed", exist_ok=True)
    out = "data/processed/ligue1_clean.csv"
    df_clean.to_csv(out, index=False)

    print(f"\n✅ {out}")
    print(f"   {len(df_clean)} matchs · {df_clean.shape[1]} colonnes")
    print(f"   Période : {df_clean['date'].min().date()} → {df_clean['date'].max().date()}")
    print(f"\nDistribution résultats :")
    print(df_clean["result"].value_counts(normalize=True).rename({"H": "Dom", "D": "Nul", "A": "Ext"}).round(3))