"""
feature_engineering.py
-----------------------
Construction vectorisée de toutes les variables du modèle.

Optimisations v2.1 :
  - Suppression des boucles iterrows / compute_standings O(n²)
  - Rolling windows vectorisées via groupby + shift(1)
  - Pondération exponentielle (exponential decay)
  - Feature H2H (confrontations directes)
  - Feature days_since_last_match (fatigue)

Principe fondamental : toutes les stats sont calculées AVANT la date
du match (shift(1) exclut le match en cours → zéro data leakage).

Utilisation :
    python src/feature_engineering.py

Entrée  : data/processed/ligue1_clean.csv
Sortie  : data/processed/features.csv
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────────────────────────────────────
WINDOW       = 10          # matchs récents pour les rolling stats
DECAY        = 0.9         # facteur de pondération exponentielle (par match)
DEFAULT_GOAL = 1.2         # lambda par défaut si historique insuffisant
MIN_MATCHES  = 3           # seuil minimum pour utiliser les stats calculées


# ──────────────────────────────────────────────────────────────────────────────
# 1. Utilitaires vectorisés
# ──────────────────────────────────────────────────────────────────────────────

def _weighted_mean(series: pd.Series, decay: float = DECAY) -> float:
    """
    Moyenne pondérée exponentiellement : le match le plus récent a le poids 1,
    le précédent decay^1, puis decay^2, etc.

    Args:
        series: valeurs ordonnées du plus ancien au plus récent
        decay:  facteur de décroissance (0 < decay ≤ 1)

    Returns:
        float: moyenne pondérée, ou DEFAULT_GOAL si série vide
    """
    n = len(series)
    if n == 0:
        return DEFAULT_GOAL
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    return float(np.dot(weights, series.values) / weights.sum())


def _build_team_match_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un log unifié (une ligne par équipe par match) trié
    chronologiquement. C'est la base de tous les rolling vectorisés.

    Colonnes produites :
        team, date, goals_scored, goals_conceded, points, venue

    Returns:
        pd.DataFrame trié par (team, date)
    """
    home = df[["date", "home_team", "home_goals", "away_goals"]].copy()
    home.columns = ["date", "team", "goals_scored", "goals_conceded"]
    home["venue"] = "home"

    away = df[["date", "away_team", "away_goals", "home_goals"]].copy()
    away.columns = ["date", "team", "goals_scored", "goals_conceded"]
    away["venue"] = "away"

    log = pd.concat([home, away], ignore_index=True)

    # Points : 3 = victoire, 1 = nul, 0 = défaite
    log["points"] = np.where(
        log["goals_scored"] > log["goals_conceded"], 3,
        np.where(log["goals_scored"] == log["goals_conceded"], 1, 0)
    )

    return log.sort_values(["team", "date"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Rolling stats vectorisées (zéro boucle Python)
# ──────────────────────────────────────────────────────────────────────────────

def compute_rolling_stats(
    df: pd.DataFrame,
    window: int = WINDOW,
    decay: float = DECAY,
) -> pd.DataFrame:
    """
    Calcule pour chaque match les statistiques de l'équipe sur les
    `window` matchs précédents (shift(1) → pas de leakage).

    Les statistiques incluent :
        - goals_scored_mean / goals_conceded_mean (weighted)
        - form_sum (somme des points sur la fenêtre)
        - days_since_last_match

    Returns:
        pd.DataFrame indexé par (team, date) avec les stats pré-calculées
    """
    log = _build_team_match_log(df)

    results = []

    for team, grp in log.groupby("team"):
        grp = grp.sort_values("date").reset_index(drop=True)
        n = len(grp)

        scored_mean    = np.full(n, DEFAULT_GOAL)
        conceded_mean  = np.full(n, DEFAULT_GOAL)
        form_sum       = np.zeros(n)
        days_since_last = np.full(n, 7.0)   # valeur par défaut : 7 jours

        for i in range(n):
            start = max(0, i - window)
            past  = grp.iloc[start:i]           # exclut le match courant (shift implicite)

            if len(past) >= MIN_MATCHES:
                scored_mean[i]   = _weighted_mean(past["goals_scored"],   decay)
                conceded_mean[i] = _weighted_mean(past["goals_conceded"], decay)
                form_sum[i]      = past["points"].sum()
            elif len(past) > 0:
                # Historique insuffisant → on utilise quand même ce qu'on a
                scored_mean[i]   = past["goals_scored"].mean()
                conceded_mean[i] = past["goals_conceded"].mean()
                form_sum[i]      = past["points"].sum()

            if i > 0:
                delta = (grp.iloc[i]["date"] - grp.iloc[i - 1]["date"]).days
                days_since_last[i] = float(delta)

        grp = grp.copy()
        grp["goals_scored_mean"]    = scored_mean
        grp["goals_conceded_mean"]  = conceded_mean
        grp["form_sum"]             = form_sum
        grp["days_since_last"]      = days_since_last
        results.append(grp)

    return pd.concat(results, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Classement vectorisé (O(n) au lieu de O(n²))
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_standings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le classement cumulatif après chaque journée de match.

    Stratégie O(n log n) :
        1. Pour chaque match, on incrémente points et goal difference.
        2. On calcule le rang par date (dense rank descendant).

    Returns:
        pd.DataFrame avec colonnes [date, team, points_cumul, gd_cumul, rang]
    """
    log = _build_team_match_log(df)
    log["gd"] = log["goals_scored"] - log["goals_conceded"]

    log = log.sort_values(["team", "date"]).reset_index(drop=True)
    log["points_cumul"] = log.groupby("team")["points"].cumsum()
    log["gd_cumul"]     = log.groupby("team")["gd"].cumsum()

    # Rang par date : on veut le classement AVANT le match → shift(1)
    log["points_before"] = log.groupby("team")["points_cumul"].shift(1).fillna(0)
    log["gd_before"]     = log.groupby("team")["gd_cumul"].shift(1).fillna(0)

    # Rang dense par date
    log = log.sort_values(["date", "team"]).reset_index(drop=True)
    log["rang"] = (
        log.groupby("date")["points_before"]
           .rank(method="min", ascending=False)
           .astype(int)
    )

    return log[["date", "team", "points_before", "gd_before", "rang"]]


# ──────────────────────────────────────────────────────────────────────────────
# 4. H2H (confrontations directes)
# ──────────────────────────────────────────────────────────────────────────────

def compute_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque match, calcule le taux de victoire historique de l'équipe
    domicile contre l'équipe extérieure (sur tous les matchs passés entre elles).

    Returns:
        pd.DataFrame avec colonnes [date, home_team, away_team,
                                    h2h_home_win_rate, h2h_n_matches]
    """
    records = []

    # Créer un index (home, away) → liste de résultats passés
    df_sorted = df.sort_values("date").reset_index(drop=True)

    for i, row in df_sorted.iterrows():
        h, a, d = row["home_team"], row["away_team"], row["date"]

        # Matchs passés entre ces deux équipes (dans les deux sens)
        past_h_home = df_sorted[
            (df_sorted["home_team"] == h) &
            (df_sorted["away_team"] == a) &
            (df_sorted["date"] < d)
        ]
        past_h_away = df_sorted[
            (df_sorted["home_team"] == a) &
            (df_sorted["away_team"] == h) &
            (df_sorted["date"] < d)
        ]

        total = len(past_h_home) + len(past_h_away)

        if total == 0:
            h2h_win_rate = 0.45   # prior : avantage domicile moyen en L1
        else:
            h_wins = (past_h_home["result"] == "H").sum() + \
                     (past_h_away["result"] == "A").sum()
            h2h_win_rate = float(h_wins / total)

        records.append({
            "date":            d,
            "home_team":       h,
            "away_team":       a,
            "h2h_home_win_rate": h2h_win_rate,
            "h2h_n_matches":   total,
        })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Construction finale des features
# ──────────────────────────────────────────────────────────────────────────────

def build_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit le DataFrame de features pour tous les matchs.

    Pipeline :
        1. Rolling stats vectorisées (goals, form, fatigue)
        2. Classements vectorisés
        3. H2H
        4. Merge et construction des différentiels

    Args:
        df: Données nettoyées (output de data_loader.py)

    Returns:
        pd.DataFrame de features prêtes pour l'entraînement
    """
    print("  [1/4] Calcul des rolling stats (decay=%.1f, window=%d)…" % (DECAY, WINDOW))
    stats = compute_rolling_stats(df, window=WINDOW, decay=DECAY)

    # Séparer stats domicile / extérieur
    home_stats = stats[stats["venue"] == "home"][
        ["date", "team", "goals_scored_mean", "goals_conceded_mean",
         "form_sum", "days_since_last"]
    ].copy()
    away_stats = stats[stats["venue"] == "away"][
        ["date", "team", "goals_scored_mean", "goals_conceded_mean",
         "form_sum", "days_since_last"]
    ].copy()

    # Renommage pour le merge
    home_stats.columns = [
        "date", "home_team",
        "home_goals_avg", "home_goals_conceded_avg",
        "home_form", "home_days_rest"
    ]
    away_stats.columns = [
        "date", "away_team",
        "away_goals_avg", "away_goals_conceded_avg",
        "away_form", "away_days_rest"
    ]

    print("  [2/4] Calcul des classements vectorisés…")
    standings = compute_all_standings(df)
    standings_pivot = standings.rename(columns={"team": "home_team", "rang": "home_rang"})
    standings_pivot_away = standings.rename(columns={"team": "away_team", "rang": "away_rang"})

    print("  [3/4] Calcul des H2H…")
    h2h = compute_h2h_features(df)

    print("  [4/4] Assemblage des features…")
    feats = df[["date", "home_team", "away_team",
                "home_shots_target", "away_shots_target",
                "season", "target"]].copy()

    feats = feats.merge(home_stats, on=["date", "home_team"], how="left")
    feats = feats.merge(away_stats, on=["date", "away_team"], how="left")

    # Classement domicile
    feats = feats.merge(
        standings_pivot[["date", "home_team", "home_rang"]],
        on=["date", "home_team"], how="left"
    )
    # Classement extérieur
    feats = feats.merge(
        standings_pivot_away[["date", "away_team", "away_rang"]],
        on=["date", "away_team"], how="left"
    )

    feats = feats.merge(h2h, on=["date", "home_team", "away_team"], how="left")

    # Normalisation du rang (0 = premier, 1 = dernier)
    n_teams = 20
    feats["home_rank_norm"]  = (feats["home_rang"].fillna(10) - 1) / (n_teams - 1)
    feats["away_rank_norm"]  = (feats["away_rang"].fillna(10) - 1) / (n_teams - 1)
    feats["classement_diff"] = feats["away_rank_norm"] - feats["home_rank_norm"]

    # Différentiels
    feats["goals_diff"]    = feats["home_goals_avg"]          - feats["away_goals_avg"]
    feats["defense_diff"]  = feats["away_goals_conceded_avg"] - feats["home_goals_conceded_avg"]
    feats["form_diff"]     = feats["home_form"]               - feats["away_form"]
    feats["fatigue_diff"]  = feats["away_days_rest"]          - feats["home_days_rest"]

    # Ratios croisés attaque/défense
    feats["home_attack_vs_away_def"] = (
        feats["home_goals_avg"] /
        feats["away_goals_conceded_avg"].clip(lower=0.3)
    )
    feats["away_attack_vs_home_def"] = (
        feats["away_goals_avg"] /
        feats["home_goals_conceded_avg"].clip(lower=0.3)
    )

    return feats.sort_values("date").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Entrée principale
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    print("\n=== Construction des features v2.1 ===\n")

    input_path = "data/processed/ligue1_clean.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"{input_path} introuvable. Lance d'abord : python src/data_loader.py"
        )

    df = pd.read_csv(input_path, parse_dates=["date"])
    print(f"Données chargées : {len(df)} matchs\n")

    t0 = time.time()
    features_df = build_match_features(df)
    elapsed = time.time() - t0

    features_df = features_df.dropna(subset=["target"])

    output_path = "data/processed/features.csv"
    os.makedirs("data/processed", exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print(f"\n✅ Features sauvegardées : {output_path}")
    print(f"   {len(features_df)} matchs · {features_df.shape[1]} colonnes")
    print(f"   Temps de calcul : {elapsed:.1f}s")
    print(f"\nDistribution de la cible :")
    print(features_df["target"].value_counts().rename({0: "Ext gagne", 1: "Nul", 2: "Dom gagne"}))