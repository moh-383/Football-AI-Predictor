"""
feature_engineering.py — Football AI Predictor v3.0
=====================================================
Changements majeurs vs v2.x :
  - h2h_draw_rate : taux de nul sur les 10 dernières confrontations directes
  - Decay rolling : _weighted_mean(decay=0.9) sur TOUTES les stats (déjà présent, renforci)
  - Anti-leakage strict : shift(1) implicite par exclusion du match courant dans la fenêtre
  - fatigue_diff : déjà présent, maintenant inclus dans FEATURE_COLS du modèle
  - home_attack_vs_away_def / away_attack_vs_home_def : ratios croisés
  - Prépare les colonnes nécessaires à Phase 5 LSTM
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Constantes globales
# ──────────────────────────────────────────────────────────────────────────────
WINDOW       = 10          # matchs récents pour rolling stats
DECAY        = 0.9         # facteur de décroissance exponentielle par match
DEFAULT_GOAL = 1.2         # lambda Poisson par défaut (historique < MIN_MATCHES)
MIN_MATCHES  = 3           # seuil minimum pour utiliser les stats calculées
H2H_WINDOW   = 10          # confrontations directes max pour H2H
N_TEAMS      = 20


# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────────────────────

def _weighted_mean(series: pd.Series, decay: float = DECAY) -> float:
    """
    Moyenne pondérée exponentiellement, du plus ancien au plus récent.
    Le match le plus récent a le poids 1, le précédent decay^1, etc.
    """
    n = len(series)
    if n == 0:
        return DEFAULT_GOAL
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    return float(np.dot(weights, series.values) / weights.sum())


def _build_team_match_log(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un log unifié (une ligne par équipe par match).
    Colonnes produites : team, date, goals_scored, goals_conceded, points, venue
    """
    home = df[["date", "home_team", "home_goals", "away_goals"]].copy()
    home.columns = ["date", "team", "goals_scored", "goals_conceded"]
    home["venue"] = "home"

    away = df[["date", "away_team", "away_goals", "home_goals"]].copy()
    away.columns = ["date", "team", "goals_scored", "goals_conceded"]
    away["venue"] = "away"

    log = pd.concat([home, away], ignore_index=True)
    log["points"] = np.where(
        log["goals_scored"] > log["goals_conceded"], 3,
        np.where(log["goals_scored"] == log["goals_conceded"], 1, 0),
    )
    return log.sort_values(["team", "date"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Rolling stats vectorisées (anti-leakage garanti : fenêtre = matchs AVANT i)
# ──────────────────────────────────────────────────────────────────────────────

def compute_rolling_stats(
    df: pd.DataFrame,
    window: int = WINDOW,
    decay: float = DECAY,
) -> pd.DataFrame:
    """
    Pour chaque match, calcule les statistiques de l'équipe sur les `window`
    matchs PRÉCÉDENTS (exclusion stricte du match courant → zéro leakage).

    Stats calculées :
        - goals_scored_mean    : moyenne pondérée (decay) des buts marqués
        - goals_conceded_mean  : moyenne pondérée des buts encaissés
        - form_sum             : somme des points sur la fenêtre
        - days_since_last      : jours depuis le dernier match (fatigue)
    """
    log     = _build_team_match_log(df)
    results = []

    for team, grp in log.groupby("team"):
        grp = grp.sort_values("date").reset_index(drop=True)
        n   = len(grp)

        scored_mean     = np.full(n, DEFAULT_GOAL)
        conceded_mean   = np.full(n, DEFAULT_GOAL)
        form_sum        = np.zeros(n)
        days_since_last = np.full(n, 7.0)

        for i in range(n):
            start = max(0, i - window)
            past  = grp.iloc[start:i]   # EXCLUSION du match i → anti-leakage

            if len(past) >= MIN_MATCHES:
                scored_mean[i]   = _weighted_mean(past["goals_scored"],   decay)
                conceded_mean[i] = _weighted_mean(past["goals_conceded"], decay)
                form_sum[i]      = past["points"].sum()
            elif len(past) > 0:
                scored_mean[i]   = past["goals_scored"].mean()
                conceded_mean[i] = past["goals_conceded"].mean()
                form_sum[i]      = past["points"].sum()
            # else : valeurs par défaut déjà assignées

            if i > 0:
                days_since_last[i] = float(
                    (grp.iloc[i]["date"] - grp.iloc[i - 1]["date"]).days
                )

        grp                       = grp.copy()
        grp["goals_scored_mean"]  = scored_mean
        grp["goals_conceded_mean"]= conceded_mean
        grp["form_sum"]           = form_sum
        grp["days_since_last"]    = days_since_last
        results.append(grp)

    return pd.concat(results, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Classement vectorisé
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_standings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classement cumulatif AVANT chaque match (shift(1) explicite).
    Retourne : date, team, points_before, gd_before, rang
    """
    log = _build_team_match_log(df)
    log["gd"] = log["goals_scored"] - log["goals_conceded"]
    log = log.sort_values(["team", "date"]).reset_index(drop=True)

    log["points_cumul"] = log.groupby("team")["points"].cumsum()
    log["gd_cumul"]     = log.groupby("team")["gd"].cumsum()

    # Points AVANT le match courant (shift anti-leakage)
    log["points_before"] = log.groupby("team")["points_cumul"].shift(1).fillna(0)
    log["gd_before"]     = log.groupby("team")["gd_cumul"].shift(1).fillna(0)

    log = log.sort_values(["date", "team"]).reset_index(drop=True)
    log["rang"] = (
        log.groupby("date")["points_before"]
           .rank(method="min", ascending=False)
           .astype(int)
    )

    return log[["date", "team", "points_before", "gd_before", "rang"]]


# ──────────────────────────────────────────────────────────────────────────────
# H2H v3.0 — win_rate + draw_rate sur les H2H_WINDOW dernières confrontations
# ──────────────────────────────────────────────────────────────────────────────

def compute_h2h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque match, calcule sur les H2H_WINDOW confrontations PASSÉES
    entre home_team et away_team (dans les deux sens) :
        - h2h_home_win_rate : taux de victoire de home_team
        - h2h_draw_rate     : taux de nul  ← NOUVEAU v3.0

    Anti-leakage : seuls les matchs STRITEMENT ANTÉRIEURS à la date courante
    sont utilisés (df["date"] < d).
    """
    records  = []
    df_sorted = df.sort_values("date").reset_index(drop=True)

    for _, row in df_sorted.iterrows():
        h, a, d = row["home_team"], row["away_team"], row["date"]

        past_hh = df_sorted[
            (df_sorted["home_team"] == h) &
            (df_sorted["away_team"] == a) &
            (df_sorted["date"] < d)
        ].tail(H2H_WINDOW)

        past_ha = df_sorted[
            (df_sorted["home_team"] == a) &
            (df_sorted["away_team"] == h) &
            (df_sorted["date"] < d)
        ].tail(H2H_WINDOW)

        total = len(past_hh) + len(past_ha)

        if total == 0:
            # Prior empirique Ligue 1 (5 saisons) : ~45% dom / ~26% nul / ~29% ext
            h2h_win_rate  = 0.45
            h2h_draw_rate = 0.26
        else:
            h_wins  = (past_hh["result"] == "H").sum() + (past_ha["result"] == "A").sum()
            draws   = (past_hh["result"] == "D").sum() + (past_ha["result"] == "D").sum()
            h2h_win_rate  = float(h_wins / total)
            h2h_draw_rate = float(draws / total)

        records.append({
            "date":           d,
            "home_team":      h,
            "away_team":      a,
            "h2h_home_win_rate": h2h_win_rate,
            "h2h_draw_rate":     h2h_draw_rate,   # NOUVEAU
            "h2h_n_matches":  total,
        })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────

def build_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit le DataFrame de features v3.0 pour tous les matchs.

    Pipeline :
        1. Rolling stats pondérées (goals, form, fatigue)
        2. Classements vectorisés
        3. H2H v3 (win_rate + draw_rate)
        4. Assemblage + différentiels + ratios croisés

    Garanties anti-leakage :
        - Fenêtre rolling exclut le match courant (past = grp.iloc[start:i])
        - Classement calculé avec points_before (shift 1)
        - H2H filtre strictement df["date"] < d
    """
    print(f"  [1/4] Rolling stats (decay={DECAY}, window={WINDOW})…")
    stats = compute_rolling_stats(df, window=WINDOW, decay=DECAY)

    home_stats = (
        stats[stats["venue"] == "home"]
        [["date", "team", "goals_scored_mean", "goals_conceded_mean",
          "form_sum", "days_since_last"]]
        .copy()
        .rename(columns={
            "team": "home_team",
            "goals_scored_mean":   "home_goals_avg",
            "goals_conceded_mean": "home_goals_conceded_avg",
            "form_sum":            "home_form",
            "days_since_last":     "home_days_rest",
        })
    )

    away_stats = (
        stats[stats["venue"] == "away"]
        [["date", "team", "goals_scored_mean", "goals_conceded_mean",
          "form_sum", "days_since_last"]]
        .copy()
        .rename(columns={
            "team": "away_team",
            "goals_scored_mean":   "away_goals_avg",
            "goals_conceded_mean": "away_goals_conceded_avg",
            "form_sum":            "away_form",
            "days_since_last":     "away_days_rest",
        })
    )

    print("  [2/4] Classements vectorisés…")
    standings = compute_all_standings(df)
    std_home  = standings.rename(columns={"team": "home_team", "rang": "home_rang"})
    std_away  = standings.rename(columns={"team": "away_team", "rang": "away_rang"})

    print("  [3/4] H2H v3 (win_rate + draw_rate)…")
    h2h = compute_h2h_features(df)

    print("  [4/4] Assemblage…")
    feats = df[[
        "date", "home_team", "away_team",
        "home_shots_target", "away_shots_target",
        "season", "target",
    ]].copy()

    feats = feats.merge(home_stats, on=["date", "home_team"], how="left")
    feats = feats.merge(away_stats, on=["date", "away_team"], how="left")

    feats = feats.merge(
        std_home[["date", "home_team", "home_rang"]],
        on=["date", "home_team"], how="left",
    )
    feats = feats.merge(
        std_away[["date", "away_team", "away_rang"]],
        on=["date", "away_team"], how="left",
    )
    feats = feats.merge(h2h, on=["date", "home_team", "away_team"], how="left")

    # Normalisation du rang [0=1er, 1=dernier]
    feats["home_rank_norm"]  = (feats["home_rang"].fillna(10) - 1) / (N_TEAMS - 1)
    feats["away_rank_norm"]  = (feats["away_rang"].fillna(10) - 1) / (N_TEAMS - 1)
    feats["classement_diff"] = feats["away_rank_norm"] - feats["home_rank_norm"]

    # Différentiels
    feats["goals_diff"]   = feats["home_goals_avg"]          - feats["away_goals_avg"]
    feats["defense_diff"] = feats["away_goals_conceded_avg"] - feats["home_goals_conceded_avg"]
    feats["form_diff"]    = feats["home_form"]               - feats["away_form"]
    feats["fatigue_diff"] = feats["away_days_rest"]          - feats["home_days_rest"]

    # Ratios croisés attaque/défense (valeur > 1 = avantage offensif)
    feats["home_attack_vs_away_def"] = (
        feats["home_goals_avg"] / feats["away_goals_conceded_avg"].clip(lower=0.3)
    )
    feats["away_attack_vs_home_def"] = (
        feats["away_goals_avg"] / feats["home_goals_conceded_avg"].clip(lower=0.3)
    )

    return feats.sort_values("date").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Feature Engineering v3.0 ===\n")

    input_path = "data/processed/ligue1_clean.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"{input_path} introuvable → lance d'abord : python src/data_loader.py"
        )

    df = pd.read_csv(input_path, parse_dates=["date"])
    print(f"Données : {len(df)} matchs ({df['date'].min().date()} → {df['date'].max().date()})\n")

    t0 = time.time()
    features_df = build_match_features(df)
    elapsed = time.time() - t0

    features_df = features_df.dropna(subset=["target"])

    os.makedirs("data/processed", exist_ok=True)
    out = "data/processed/features.csv"
    features_df.to_csv(out, index=False)

    print(f"\n✅ {out}")
    print(f"   {len(features_df)} matchs · {features_df.shape[1]} colonnes · {elapsed:.1f}s")
    print("\nDistribution cible :")
    print(
        features_df["target"]
        .value_counts()
        .rename({0: "Ext gagne", 1: "Nul", 2: "Dom gagne"})
        .to_string()
    )
    print()

    # Vérification des nouvelles colonnes v3
    for col in ["h2h_draw_rate", "fatigue_diff", "home_attack_vs_away_def"]:
        null_pct = features_df[col].isna().mean()
        print(f"  {col}: {null_pct:.1%} NaN")