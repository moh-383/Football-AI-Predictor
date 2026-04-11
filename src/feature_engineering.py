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


def compute_standings(df, date):
    """
    Calcule le classement de toutes les équipes AVANT une date donnée,
    à partir des matchs joués jusqu'à cette date.

    Retourne un dict : {nom_equipe: rang}  (1 = premier, 20 = dernier)
    """
    past = df[df['date'] < date].copy()

    if len(past) == 0:
        return {}

    teams = pd.concat([past['home_team'], past['away_team']]).unique()
    points = {t: 0 for t in teams}
    gd     = {t: 0 for t in teams}  # goal difference

    for _, row in past.iterrows():
        h, a   = row['home_team'], row['away_team']
        hg, ag = row['home_goals'], row['away_goals']

        gd[h] += hg - ag
        gd[a] += ag - hg

        if hg > ag:
            points[h] += 3
        elif hg == ag:
            points[h] += 1
            points[a] += 1
        else:
            points[a] += 3

    standings = pd.DataFrame({'points': points, 'gd': gd})
    standings = standings.sort_values(
        ['points', 'gd'], ascending=False
    )
    standings['rang'] = range(1, len(standings) + 1)

    return standings['rang'].to_dict()

def get_h2h_stats(df, home_team, away_team, date, n=10):
    """
    Calcule les stats des N derniers matchs entre deux équipes.
    Retourne : win_rate_home, avg_goals_home, avg_goals_away
    """

    # Filtrer : matchs entre ces deux équipes AVANT la date du match
    mask = (
        (df["date"] < date) &
        ((
            (df["home_team"] == home_team) & (df["away_team"] == away_team)
        ) | (
            (df["home_team"] == away_team) & (df["away_team"] == home_team)
        ))
    )

    h2h_matches = df[mask].tail(n)  # On prend les N derniers seulement

    # Si aucun match h2h trouvé → retourner des valeurs neutres
    if len(h2h_matches) == 0:
        return {
            "h2h_win_rate_home":  0.33,  # 1/3 par défaut (équiprobable)
            "h2h_avg_goals_home": 1.5,   # moyenne Ligue 1
            "h2h_avg_goals_away": 1.0,
            "h2h_n_matches":      0      # aucun match trouvé
        }

    # Compter les victoires de home_team dans ces matchs h2h
    home_wins = 0

    for _, row in h2h_matches.iterrows():
        if row["home_team"] == home_team:
            # home_team jouait à domicile dans ce match
            if row["home_goals"] > row["away_goals"]:
                home_wins += 1
        else:
            # home_team jouait à l'extérieur dans ce match
            if row["away_goals"] > row["home_goals"]:
                home_wins += 1

    # Taux de victoire de home_team sur les h2h
    win_rate_home = home_wins / len(h2h_matches)

    # Moyenne de buts marqués par home_team dans les h2h
    home_goals_list = []
    away_goals_list = []

    for _, row in h2h_matches.iterrows():
        if row["home_team"] == home_team:
            home_goals_list.append(row["home_goals"])
            away_goals_list.append(row["away_goals"])
        else:
            # Les rôles sont inversés dans ce match
            home_goals_list.append(row["away_goals"])
            away_goals_list.append(row["home_goals"])

    avg_goals_home = sum(home_goals_list) / len(home_goals_list)
    avg_goals_away = sum(away_goals_list) / len(away_goals_list)

    return {
        "h2h_win_rate_home":  win_rate_home,
        "h2h_avg_goals_home": avg_goals_home,
        "h2h_avg_goals_away": avg_goals_away,
        "h2h_n_matches":      len(h2h_matches)
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

        # Classement dynamique à la date du match
        standings = compute_standings(df, match['date'])
        home_rank = standings.get(match['home_team'], 10)
        away_rank = standings.get(match['away_team'], 10)
        n_teams   = len(standings) if standings else 20

        # Normalisation : 0 = premier, 1 = dernier
        home_rank_norm = (home_rank - 1) / max(n_teams - 1, 1)
        away_rank_norm = (away_rank - 1) / max(n_teams - 1, 1)

        h2h = get_h2h_stats(df, match["home_team"], match["away_team"], match["date"])

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

#---Nouvelle feature h2h
 "h2h_win_rate_home":  h2h["h2h_win_rate_home"],
            "h2h_avg_goals_home": h2h["h2h_avg_goals_home"],
            "h2h_avg_goals_away": h2h["h2h_avg_goals_away"],

            "target": match["target"]
            # --- Tier 1 : Classement ---
            'classement_diff':     away_rank_norm - home_rank_norm,
            'home_rank_norm':      home_rank_norm,
            'away_rank_norm':      away_rank_norm,


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

           
            # --- Tier 2 : Ratio croise attaque/defense ---
            'home_attack_vs_away_def': (
                home_stats['goals_scored_mean'] /
                max(away_stats['goals_conceded_mean'], 0.3)
            ),
            'away_attack_vs_home_def': (
                away_stats['goals_scored_mean'] /
                max(home_stats['goals_conceded_mean'], 0.3)
            ),
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
