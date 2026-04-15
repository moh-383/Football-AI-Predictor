"""
data_loader.py
--------------
Téléchargement et nettoyage des données brutes de matchs de football.

Source : football-data.co.uk (CSV gratuits, toutes ligues européennes)

Utilisation :
    python src/data_loader.py

Sortie :
    data/raw/ligue1_XXXX.csv        — CSV bruts par saison
    data/processed/ligue1_clean.csv — Données nettoyées et triées par date
"""

import pandas as pd
import requests
import os


def download_ligue1_data(seasons):
    """
    Télécharge les données Ligue 1 pour les saisons demandées.

    Args:
        seasons (list[str]): Codes de saisons, ex: ["2122", "2223", "2324"]
            Format : deux derniers chiffres de l'année de début + fin
            Exemple : "2324" = saison 2023-2024

    Returns:
        pd.DataFrame: Toutes les saisons concaténées
    """
    os.makedirs("data/raw", exist_ok=True)
    base_url = "https://www.football-data.co.uk/mmz4281/{season}/F1.csv"
    all_dfs = []

    for season in seasons:
        url = base_url.format(season=season)
        response = requests.get(url)

        if response.status_code == 200:
            filepath = f"data/raw/ligue1_{season}.csv"
            with open(filepath, "wb") as f:
                f.write(response.content)

            df = pd.read_csv(filepath, encoding="latin1")
            df["season"] = season
            all_dfs.append(df)
            print(f"  ✅ Saison {season} : {len(df)} matchs téléchargés")
        else:
            print(f"  ❌ Saison {season} : HTTP {response.status_code}")

    if not all_dfs:
        raise RuntimeError("Aucune donnée téléchargée. Vérifiez votre connexion.")

    return pd.concat(all_dfs, ignore_index=True)


def clean_data(df):
    """
    Nettoie et standardise les données brutes.

    IMPORTANT : les données sont triées par date chronologique pour
    éviter toute fuite temporelle (data leakage) lors du feature engineering.

    Args:
        df (pd.DataFrame): DataFrame brut issu de download_ligue1_data()

    Returns:
        pd.DataFrame: Données nettoyées avec colonnes renommées et typées
    """
    cols = [
        "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HS", "AS", "HST", "AST", "HC", "AC",
        "HF", "AF", "HY", "AY", "HR", "AR", "season"
    ]

    # Garder uniquement les colonnes disponibles (certaines peuvent manquer)
    cols_available = [c for c in cols if c in df.columns]
    df = df[cols_available].copy()

    rename_map = {
        "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team",
        "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
        "HS": "home_shots", "AS": "away_shots",
        "HST": "home_shots_target", "AST": "away_shots_target",
        "HC": "home_corners", "AC": "away_corners",
        "HF": "home_fouls", "AF": "away_fouls",
        "HY": "home_yellow", "AY": "away_yellow",
        "HR": "home_red", "AR": "away_red",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    # Encodage de la variable cible : 0=ext gagne, 1=nul, 2=dom gagne
    df["target"] = df["result"].map({"H": 2, "D": 1, "A": 0})

    df = df.dropna(subset=["home_goals", "away_goals", "result"])

    # CRUCIAL : tri chronologique strict
    df = df.sort_values("date").reset_index(drop=True)

    return df


if __name__ == "__main__":
    print("\n=== Téléchargement des données Ligue 1 ===\n")
    seasons = ["1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324"]
    df_raw = download_ligue1_data(seasons)

    print(f"\nTotal brut : {len(df_raw)} matchs, {df_raw.shape[1]} colonnes")
    print("Distribution des résultats (brut) :")
    print(df_raw["FTR"].value_counts(normalize=True).round(3))

    print("\nNettoyage des données...")
    df_clean = clean_data(df_raw)

    os.makedirs("data/processed", exist_ok=True)
    df_clean.to_csv("data/processed/ligue1_clean.csv", index=False)
    print(f"\n✅ Données nettoyées sauvegardées : data/processed/ligue1_clean.csv")
    print(f"   {len(df_clean)} matchs · colonnes : {list(df_clean.columns)}")
