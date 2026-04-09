"""
feature_engineering.py
-----------------------
Construction de toutes les variables (features) utilisées par le modèle.

Principe fondamental : toutes les statistiques sont calculées AVANT la date
du match pour éviter la fuite temporelle (data leakage). On utilise shift(1)
sur les rolling windows pour exclure le match en cours.

Utilisation :
    python src/feature_engineering.py

Entrée :
    data/processed/ligue1_clean.csv

Sortie :
    data/processed/features.csv
"""

import pandas as pd
import numpy as np
import os


def get_team_stats_before(df, team, date, venue="both", window=10):
    """
    Calcule les statistiques d'une équipe AVANT une date donnée.

    Args:
        df (pd.DataFrame): DataFrame des matchs historiques (trié par date)
        team (str): Nom de l'équipe
        date (pd.Timestamp): Date limite (exclusive)
        venue (str): "home", "away", ou "both"
        window (int): Nombre de matchs récents à considérer

    Returns:
        dict: Statistiques calculées (goals_scored_mean, goals_conceded_mean, etc.)
    """
    if venue == "home":
        mask = (df["home_team"] == team) & (df["date"] < date)
        goals_scored   = df.loc[mask, "home_goals"]
        goals_conceded = df.loc[mask, "away_goals"]
    elif venue == "away":
        mask = (df["away_team"] == team) & (df["date"] < date)
        goals_scored   = df.loc[mask, "away_goals"]
        goals_conceded = df.loc[mask, "home_goals"]
    else:  # both
        mask_h = (df["home_team"] == team) & (df["date"] < date)
        mask_a = (df["away_team"] == team) & (df["date"] < date)
        goals_scored = pd.concat([
            df.loc[mask_h, "home_goals"],
            df.loc[mask_a, "away_goals"]
        ]).sort_index()
        goals_conceded = pd.concat([
            df.loc[mask_h, "away_goals"],
            df.loc[mask_a, "home_goals"]
        ]).sort_index()

    recent_scored    = goals_scored.tail(window)
    recent_conceded  = goals_conceded.tail(window)

    # Valeur par défaut si pas assez de matchs historiques
    default_goals = 1.2

    return {
        "goals_scored_mean":   recent_scored.mean()   if len(recent_scored) > 2   else default_goals,
        "goals_conceded_mean": recent_conceded.mean() if len(recent_conceded) > 2 else default_goals,
        "points_sum":          recent_scored.apply(lambda g: 1 if g > 0 else 0).sum(),
        "n_matches":           len(recent_scored),
    }


def build_match_features(df):
    """
    Construit le DataFrame de features pour tous les matchs du dataset.

    Chaque ligne du résultat correspond à un match, avec toutes les variables
    calculées à partir des données historiques antérieures à ce match.

    Args:
        df (pd.DataFrame): Données nettoyées (output de data_loader.py)

    Returns:
        pd.DataFrame: Features prêtes pour l'entraînement du modèle
    """
    features = []
    total = len(df)

    for idx, match in df.iterrows():
        home_stats  = get_team_stats_before(df, match["home_team"], match["date"], "home")
        away_stats  = get_team_stats_before(df, match["away_team"], match["date"], "away")
        all_home    = get_team_stats_before(df, match["home_team"], match["date"], "both")
        all_away    = get_team_stats_before(df, match["away_team"], match["date"], "both")

        row = {
            # --- Tier 1 : Offensif ---
            "home_goals_avg":          home_stats["goals_scored_mean"],
            "away_goals_avg":          away_stats["goals_scored_mean"],

            # --- Tier 1 : Défensif ---
            "home_goals_conceded_avg": home_stats["goals_conceded_mean"],
            "away_goals_conceded_avg": away_stats["goals_conceded_mean"],

            # --- Tier 1 : Forme récente ---
            "home_form":               all_home["points_sum"],
            "away_form":               all_away["points_sum"],

            # --- Tier 1 : Différentiels (très informatifs) ---
            "goals_diff":              home_stats["goals_scored_mean"]   - away_stats["goals_scored_mean"],
            "defense_diff":            away_stats["goals_conceded_mean"] - home_stats["goals_conceded_mean"],
            "form_diff":               all_home["points_sum"]            - all_away["points_sum"],

            # --- Tier 2 : Tirs cadrés (si disponibles) ---
            "home_shots_target":       match.get("home_shots_target", np.nan),
            "away_shots_target":       match.get("away_shots_target", np.nan),

            # --- Métadonnées (non utilisées comme features) ---
            "date":                    match["date"],
            "home_team":               match["home_team"],
            "away_team":               match["away_team"],
            "season":                  match.get("season", ""),

            # --- Variable cible ---
            "target":                  match["target"],
        }
        features.append(row)

        if idx % 200 == 0:
            pct = (idx / total) * 100
            print(f"  Traitement : {idx}/{total} matchs ({pct:.0f}%)")

    return pd.DataFrame(features)


if __name__ == "__main__":
    print("\n=== Construction des features ===\n")

    input_path = "data/processed/ligue1_clean.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"{input_path} introuvable. Lance d'abord : python src/data_loader.py"
        )

    df = pd.read_csv(input_path, parse_dates=["date"])
    print(f"Données chargées : {len(df)} matchs\n")

    features_df = build_match_features(df)

    # Supprimer les lignes avec target manquant
    features_df = features_df.dropna(subset=["target"])

    output_path = "data/processed/features.csv"
    features_df.to_csv(output_path, index=False)

    print(f"\n✅ Features sauvegardées : {output_path}")
    print(f"   {len(features_df)} matchs · {features_df.shape[1]} colonnes")
    print(f"\nDistribution de la cible :")
    print(features_df["target"].value_counts().rename({0: "Ext gagne", 1: "Nul", 2: "Dom gagne"}))
