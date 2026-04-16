"""
predictor.py — Football AI Predictor v3.0
==========================================
Changements vs v2.x :
  - Seuil asymétrique nul : si P(Nul) > NUL_THRESHOLD (0.22), signale le match
    comme "Nul probable" même s'il n'est pas la classe majoritaire
  - h2h_draw_rate intégré dans le vecteur de features
  - Cohérence stricte avec FEATURE_COLS de model.py v3.0
"""

from __future__ import annotations

import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes — identiques à feature_engineering.py et model.py
# ──────────────────────────────────────────────────────────────────────────────
WINDOW        = 10
DECAY         = 0.9
DEFAULT_GOAL  = 1.2
MIN_MATCHES   = 3
N_TEAMS       = 20
H2H_WINDOW    = 10

# Seuil asymétrique v3.0 : signal nul même sans être la classe majoritaire
NUL_THRESHOLD = 0.22

# Constantes de normalisation (mirror de add_draw_features dans model.py)
GOAL_DIFF_SCALE = 3.0
MAX_FORM        = 30.0


# ──────────────────────────────────────────────────────────────────────────────
# Rolling stats — mirror de feature_engineering.py
# ──────────────────────────────────────────────────────────────────────────────

def _weighted_mean(series: pd.Series, decay: float = DECAY) -> float:
    n = len(series)
    if n == 0:
        return DEFAULT_GOAL
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    return float(np.dot(weights, series.values) / weights.sum())


def get_team_rolling_stats(
    df: pd.DataFrame, team: str, date: pd.Timestamp, venue: str = "both"
) -> dict:
    """
    Calcule les stats rolling d'une équipe AVANT une date donnée.
    venue : "home", "away", ou "both"
    """
    if venue == "home":
        mask     = (df["home_team"] == team) & (df["date"] < date)
        scored   = df.loc[mask, "home_goals"]
        conceded = df.loc[mask, "away_goals"]
        dates_c  = df.loc[mask, "date"]
    elif venue == "away":
        mask     = (df["away_team"] == team) & (df["date"] < date)
        scored   = df.loc[mask, "away_goals"]
        conceded = df.loc[mask, "home_goals"]
        dates_c  = df.loc[mask, "date"]
    else:
        mh = (df["home_team"] == team) & (df["date"] < date)
        ma = (df["away_team"] == team) & (df["date"] < date)
        scored = pd.concat([
            df.loc[mh, "home_goals"], df.loc[ma, "away_goals"]
        ]).sort_index()
        conceded = pd.concat([
            df.loc[mh, "away_goals"], df.loc[ma, "home_goals"]
        ]).sort_index()
        dates_c = pd.concat([
            df.loc[mh, "date"], df.loc[ma, "date"]
        ]).sort_values()

    r_scored   = scored.tail(WINDOW)
    r_conceded = conceded.tail(WINDOW)
    r_dates    = dates_c.sort_values().tail(WINDOW)
    n = len(r_scored)

    if n >= MIN_MATCHES:
        g_scored   = _weighted_mean(r_scored)
        g_conceded = _weighted_mean(r_conceded)
    elif n > 0:
        g_scored   = r_scored.mean()
        g_conceded = r_conceded.mean()
    else:
        g_scored   = DEFAULT_GOAL
        g_conceded = DEFAULT_GOAL

    form_sum = 0.0
    if n > 0:
        pts = (r_scored.values > r_conceded.values).astype(int) * 3 + \
              (r_scored.values == r_conceded.values).astype(int)
        form_sum = float(pts.sum())

    days_since = 7.0
    if len(r_dates) > 0:
        days_since = float((date - r_dates.iloc[-1]).days)

    return {
        "goals_scored_mean":   g_scored,
        "goals_conceded_mean": g_conceded,
        "form_sum":            form_sum,
        "days_since_last":     days_since,
        "n_matches":           n,
    }


def get_h2h_stats(
    df: pd.DataFrame, home_team: str, away_team: str, date: pd.Timestamp
) -> dict:
    """
    H2H v3.0 : win_rate + draw_rate sur les H2H_WINDOW dernières confrontations.
    """
    past_hh = df[
        (df["home_team"] == home_team) &
        (df["away_team"] == away_team) &
        (df["date"] < date)
    ].tail(H2H_WINDOW)

    past_ha = df[
        (df["home_team"] == away_team) &
        (df["away_team"] == home_team) &
        (df["date"] < date)
    ].tail(H2H_WINDOW)

    total = len(past_hh) + len(past_ha)

    if total == 0:
        return {"h2h_home_win_rate": 0.45, "h2h_draw_rate": 0.26}

    h_wins = (past_hh["result"] == "H").sum() + (past_ha["result"] == "A").sum()
    draws  = (past_hh["result"] == "D").sum() + (past_ha["result"] == "D").sum()

    return {
        "h2h_home_win_rate": float(h_wins / total),
        "h2h_draw_rate":     float(draws / total),
    }


def get_rank_norm(df: pd.DataFrame, team: str, date: pd.Timestamp) -> float:
    """Rang normalisé de l'équipe AVANT la date [0=1er, 1=dernier]."""
    past = df[df["date"] < date].copy()
    if len(past) == 0:
        return 0.47

    teams  = pd.concat([past["home_team"], past["away_team"]]).unique()
    points = {t: 0 for t in teams}
    gd     = {t: 0 for t in teams}

    for _, row in past.iterrows():
        h, a = row["home_team"], row["away_team"]
        hg, ag = row["home_goals"], row["away_goals"]
        gd[h] += hg - ag
        gd[a] += ag - hg
        if hg > ag:
            points[h] += 3
        elif hg == ag:
            points[h] += 1
            points[a] += 1
        else:
            points[a] += 3

    standings = pd.DataFrame({"points": points, "gd": gd})
    standings = standings.sort_values(["points", "gd"], ascending=False)
    standings["rang"] = range(1, len(standings) + 1)
    rang = standings.loc[team, "rang"] if team in standings.index else 10
    return (rang - 1) / (N_TEAMS - 1)


# ──────────────────────────────────────────────────────────────────────────────
# Features dérivées — mirror de model.py add_draw_features
# ──────────────────────────────────────────────────────────────────────────────

def compute_derived_features(goals_diff: float, home_form: float, away_form: float) -> dict:
    strength_symmetry = 1.0 - min(abs(goals_diff) / GOAL_DIFF_SCALE, 1.0)
    draw_prior        = 1.0 - (
        (min(home_form, MAX_FORM) + min(away_form, MAX_FORM)) / (2 * MAX_FORM)
    )
    return {"strength_symmetry": strength_symmetry, "draw_prior": draw_prior}


# ──────────────────────────────────────────────────────────────────────────────
# Prédiction principale
# ──────────────────────────────────────────────────────────────────────────────

def predict_match(
    home_team: str,
    away_team: str,
    df_historical: pd.DataFrame,
    date=None,
) -> dict:
    """
    Prédit les probabilités d'issue avec seuil asymétrique nul (v3.0).

    Returns:
        dict avec prob_home_win, prob_draw, prob_away_win,
              expected_goals_home, expected_goals_away,
              most_likely_score, nul_alert
    """
    # ── Chargement artefacts ──────────────────────────────────────────────────
    model_path   = "models/xgb_model.pkl"
    medians_path = "models/feature_medians.pkl"
    feature_path = "models/feature_names.txt"

    for p in [model_path, feature_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} introuvable → lance d'abord python src/model.py")

    model         = joblib.load(model_path)
    feature_cols  = [l.strip() for l in open(feature_path) if l.strip()]
    medians       = joblib.load(medians_path) if os.path.exists(medians_path) else {}

    # ── Date ──────────────────────────────────────────────────────────────────
    date = pd.Timestamp(date) if date is not None else pd.Timestamp.today()

    # ── Stats rolling ─────────────────────────────────────────────────────────
    hs = get_team_rolling_stats(df_historical, home_team, date, "home")
    as_ = get_team_rolling_stats(df_historical, away_team, date, "away")

    # ── Classement ────────────────────────────────────────────────────────────
    h_rank = get_rank_norm(df_historical, home_team, date)
    a_rank = get_rank_norm(df_historical, away_team, date)
    classement_diff = a_rank - h_rank

    # ── H2H v3.0 ──────────────────────────────────────────────────────────────
    h2h = get_h2h_stats(df_historical, home_team, away_team, date)

    # ── Différentiels ─────────────────────────────────────────────────────────
    goals_diff   = hs["goals_scored_mean"]   - as_["goals_scored_mean"]
    defense_diff = as_["goals_conceded_mean"] - hs["goals_conceded_mean"]
    form_diff    = hs["form_sum"]             - as_["form_sum"]
    fatigue_diff = as_["days_since_last"]     - hs["days_since_last"]

    # ── Features dérivées ─────────────────────────────────────────────────────
    derived = compute_derived_features(goals_diff, hs["form_sum"], as_["form_sum"])

    # ── Vecteur complet ───────────────────────────────────────────────────────
    feature_dict = {
        "home_goals_avg":           hs["goals_scored_mean"],
        "away_goals_avg":           as_["goals_scored_mean"],
        "home_goals_conceded_avg":  hs["goals_conceded_mean"],
        "away_goals_conceded_avg":  as_["goals_conceded_mean"],
        "home_form":                hs["form_sum"],
        "away_form":                as_["form_sum"],
        "goals_diff":               goals_diff,
        "defense_diff":             defense_diff,
        "form_diff":                form_diff,
        "classement_diff":          classement_diff,
        "home_rank_norm":           h_rank,
        "away_rank_norm":           a_rank,
        "home_shots_target":        np.nan,   # indisponible avant le match
        "away_shots_target":        np.nan,
        "h2h_home_win_rate":        h2h["h2h_home_win_rate"],
        "h2h_draw_rate":            h2h["h2h_draw_rate"],      # NOUVEAU v3.0
        "fatigue_diff":             fatigue_diff,
        "strength_symmetry":        derived["strength_symmetry"],
        "draw_prior":               derived["draw_prior"],
        "home_attack_vs_away_def":  hs["goals_scored_mean"] / max(as_["goals_conceded_mean"], 0.3),
        "away_attack_vs_home_def":  as_["goals_scored_mean"] / max(hs["goals_conceded_mean"], 0.3),
    }

    # Vérification cohérence avec feature_names.txt
    missing = [f for f in feature_cols if f not in feature_dict]
    if missing:
        raise KeyError(
            f"\n❌ Features manquantes dans predictor.py feature_dict :\n"
            f"   {missing}\n   → Ajoute-les dans feature_dict."
        )

    # ── Construction de X dans l'ordre exact ──────────────────────────────────
    X = np.array([[feature_dict[f] for f in feature_cols]], dtype=float)

    for i, col in enumerate(feature_cols):
        if np.isnan(X[0, i]):
            X[0, i] = float(medians.get(col, 0.0)) if isinstance(medians, dict) \
                      else float(medians.get(col, 0.0))

    # ── Prédiction ────────────────────────────────────────────────────────────
    probas = model.predict_proba(X)[0]   # [p_ext, p_nul, p_dom]

    # ── Seuil asymétrique nul v3.0 ────────────────────────────────────────────
    # Si P(Nul) > NUL_THRESHOLD, on le signale même si ce n'est pas la classe max
    nul_alert = bool(probas[1] >= NUL_THRESHOLD)

    # ── Score le plus probable via Poisson ────────────────────────────────────
    buts    = np.arange(0, 7)
    matrice = np.outer(
        poisson.pmf(buts, hs["goals_scored_mean"]),
        poisson.pmf(buts, as_["goals_scored_mean"]),
    )
    best_score = np.unravel_index(matrice.argmax(), matrice.shape)

    result = {
        "match":               f"{home_team} vs {away_team}",
        "prob_home_win":       float(probas[2]),
        "prob_draw":           float(probas[1]),
        "prob_away_win":       float(probas[0]),
        "expected_goals_home": round(hs["goals_scored_mean"], 2),
        "expected_goals_away": round(as_["goals_scored_mean"], 2),
        "most_likely_score":   f"{best_score[0]} - {best_score[1]}",
        "nul_alert":           nul_alert,
        "h2h_draw_rate":       round(h2h["h2h_draw_rate"], 3),
    }

    _print_result(result, home_team, away_team)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Affichage
# ──────────────────────────────────────────────────────────────────────────────

def _print_result(result: dict, home_team: str, away_team: str) -> None:
    print(f"\n{'═'*54}")
    print(f"  ⚽  {home_team} vs {away_team}")
    print(f"{'═'*54}")

    items = [
        (f"Victoire {home_team}", result["prob_home_win"]),
        ("Match nul",             result["prob_draw"]),
        (f"Victoire {away_team}", result["prob_away_win"]),
    ]
    for label, prob in items:
        bar = "█" * int(prob * 32)
        print(f"  {label:<26} {prob:5.1%}  {bar}")

    if result["nul_alert"]:
        print(f"\n  ⚠️  ALERTE NUL : P(Nul)={result['prob_draw']:.1%} ≥ {NUL_THRESHOLD:.0%}")
        print(f"     H2H draw rate historique : {result['h2h_draw_rate']:.0%}")

    print(f"\n  Score probable  : {result['most_likely_score']}")
    print(f"  xG dom / ext    : {result['expected_goals_home']} / {result['expected_goals_away']}")
    print(f"{'═'*54}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédire un match de Ligue 1 (v3.0)")
    parser.add_argument("--home", required=True, help="Équipe domicile")
    parser.add_argument("--away", required=True, help="Équipe extérieur")
    parser.add_argument("--date", default=None,  help="Date YYYY-MM-DD (défaut: aujourd'hui)")
    parser.add_argument("--data", default="data/processed/ligue1_clean.csv")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"{args.data} introuvable → lance data_loader.py")

    df = pd.read_csv(args.data, parse_dates=["date"])
    predict_match(args.home, args.away, df, date=args.date)