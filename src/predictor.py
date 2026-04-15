"""
predictor.py
------------
Interface principale de prédiction d'un match.

Utilisation en ligne de commande :
    python src/predictor.py --home "PSG" --away "Marseille"
    python src/predictor.py --home "Lyon" --away "Nice" --date "2025-05-10"

Utilisation comme module Python :
    from predictor import predict_match
    result = predict_match("PSG", "Marseille", df_historical)
"""

import argparse
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Constantes — doivent être identiques à feature_engineering.py et model.py
# ──────────────────────────────────────────────────────────────────────────────
WINDOW        = 10
DECAY         = 0.9
DEFAULT_GOAL  = 1.2
MIN_MATCHES   = 3
N_TEAMS       = 20

# Constantes de normalisation — identiques à add_draw_features() dans model.py
GOAL_DIFF_SCALE = 3.0
MAX_FORM        = 30.0
FORM_DIFF_SCALE = 30.0


# ──────────────────────────────────────────────────────────────────────────────
# Calcul des stats rolling (miroir de feature_engineering.py)
# ──────────────────────────────────────────────────────────────────────────────

def _weighted_mean(series: pd.Series, decay: float = DECAY) -> float:
    """Moyenne pondérée exponentiellement, du plus ancien au plus récent."""
    n = len(series)
    if n == 0:
        return DEFAULT_GOAL
    weights = np.array([decay ** (n - 1 - i) for i in range(n)])
    return float(np.dot(weights, series.values) / weights.sum())


def get_team_rolling_stats(df: pd.DataFrame, team: str, date: pd.Timestamp, venue: str = "both") -> dict:
    """
    Calcule les stats rolling d'une équipe AVANT une date donnée.

    Args:
        df:     historique complet des matchs (trié par date)
        team:   nom de l'équipe
        date:   date limite exclusive
        venue:  "home", "away", ou "both"

    Returns:
        dict avec goals_scored_mean, goals_conceded_mean, form_sum,
        days_since_last, n_matches
    """
    if venue == "home":
        mask = (df["home_team"] == team) & (df["date"] < date)
        scored    = df.loc[mask, "home_goals"]
        conceded  = df.loc[mask, "away_goals"]
        dates_col = df.loc[mask, "date"]
    elif venue == "away":
        mask = (df["away_team"] == team) & (df["date"] < date)
        scored    = df.loc[mask, "away_goals"]
        conceded  = df.loc[mask, "home_goals"]
        dates_col = df.loc[mask, "date"]
    else:  # both
        mask_h = (df["home_team"] == team) & (df["date"] < date)
        mask_a = (df["away_team"] == team) & (df["date"] < date)

        scored = pd.concat([
            df.loc[mask_h, "home_goals"],
            df.loc[mask_a, "away_goals"],
        ]).sort_index()
        conceded = pd.concat([
            df.loc[mask_h, "away_goals"],
            df.loc[mask_a, "home_goals"],
        ]).sort_index()
        dates_col = pd.concat([
            df.loc[mask_h, "date"],
            df.loc[mask_a, "date"],
        ]).sort_values()

    # Garder uniquement les WINDOW derniers matchs
    recent_scored   = scored.tail(WINDOW)
    recent_conceded = conceded.tail(WINDOW)
    recent_dates    = dates_col.sort_values().tail(WINDOW)
    n = len(recent_scored)

    if n >= MIN_MATCHES:
        goals_scored_mean   = _weighted_mean(recent_scored)
        goals_conceded_mean = _weighted_mean(recent_conceded)
    elif n > 0:
        goals_scored_mean   = recent_scored.mean()
        goals_conceded_mean = recent_conceded.mean()
    else:
        goals_scored_mean   = DEFAULT_GOAL
        goals_conceded_mean = DEFAULT_GOAL

    # Forme : somme des points sur les matchs récents
    if n > 0:
        results = (recent_scored.values > recent_conceded.values).astype(int) * 3 + \
                  (recent_scored.values == recent_conceded.values).astype(int)
        form_sum = float(results.sum())
    else:
        form_sum = 0.0

    # Jours depuis le dernier match
    if len(recent_dates) > 0:
        last_match_date = recent_dates.iloc[-1]
        days_since_last = float((date - last_match_date).days)
    else:
        days_since_last = 7.0   # valeur par défaut

    return {
        "goals_scored_mean":   goals_scored_mean,
        "goals_conceded_mean": goals_conceded_mean,
        "form_sum":            form_sum,
        "days_since_last":     days_since_last,
        "n_matches":           n,
    }


def get_h2h_win_rate(df: pd.DataFrame, home_team: str, away_team: str, date: pd.Timestamp) -> float:
    """
    Calcule le taux de victoire historique de home_team face à away_team
    AVANT la date donnée (dans les deux sens de confrontation).

    Miroir exact de compute_h2h_features() dans feature_engineering.py.
    """
    past_h_home = df[
        (df["home_team"] == home_team) &
        (df["away_team"] == away_team) &
        (df["date"] < date)
    ]
    past_h_away = df[
        (df["home_team"] == away_team) &
        (df["away_team"] == home_team) &
        (df["date"] < date)
    ]

    total = len(past_h_home) + len(past_h_away)
    if total == 0:
        return 0.45   # prior : avantage domicile moyen en L1

    h_wins = (past_h_home["result"] == "H").sum() + \
             (past_h_away["result"] == "A").sum()
    return float(h_wins / total)


def get_rank_norm(df: pd.DataFrame, team: str, date: pd.Timestamp) -> float:
    """
    Calcule le rang normalisé d'une équipe AVANT la date donnée.
    0 = premier, 1 = dernier. Miroir de compute_all_standings().
    """
    past = df[df["date"] < date].copy()
    if len(past) == 0:
        return 0.47   # médiane (rang 10 sur 20)

    teams = pd.concat([past["home_team"], past["away_team"]]).unique()
    points = {t: 0 for t in teams}
    gd     = {t: 0 for t in teams}

    for _, row in past.iterrows():
        h, a   = row["home_team"], row["away_team"]
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
# Features dérivées — miroir exact de add_draw_features() dans model.py
# ──────────────────────────────────────────────────────────────────────────────

def compute_derived_features(goals_diff: float, home_form: float, away_form: float,
                              classement_diff: float, form_diff: float) -> dict:
    """
    Calcule strength_symmetry et draw_prior.
    Constantes identiques à add_draw_features() dans model.py.
    """
    strength_symmetry = float(
        1.0 - min(abs(goals_diff) / GOAL_DIFF_SCALE, 1.0)
    )
    draw_prior = float(
        1.0 - (
            (min(home_form, MAX_FORM) + min(away_form, MAX_FORM))
            / (2 * MAX_FORM)
        )
    )
    return {
        "strength_symmetry": strength_symmetry,
        "draw_prior":        draw_prior,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Prédiction principale
# ──────────────────────────────────────────────────────────────────────────────

def predict_match(home_team: str, away_team: str, df_historical: pd.DataFrame,
                  date=None) -> dict:
    """
    Prédit les probabilités d'issue d'un match.

    Args:
        home_team:      Nom de l'équipe à domicile
        away_team:      Nom de l'équipe à l'extérieur
        df_historical:  Historique des matchs (colonnes : date, home_team,
                        away_team, home_goals, away_goals, result, …)
        date:           Date du match (str "YYYY-MM-DD" ou Timestamp).
                        Défaut : aujourd'hui.

    Returns:
        dict avec prob_home_win, prob_draw, prob_away_win,
              expected_goals_home, expected_goals_away, most_likely_score
    """
    # ── Chargement des artefacts ──────────────────────────────────────────────
    model_path   = "models/xgb_model.pkl"
    medians_path = "models/feature_medians.pkl"
    feature_path = "models/feature_names.txt"

    for path in [model_path, feature_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} introuvable. Lance d'abord : python src/model.py"
            )

    model       = joblib.load(model_path)
    feature_cols = [l.strip() for l in open(feature_path) if l.strip()]

    # Médianes d'imputation (sauvegardées à l'entraînement)
    if os.path.exists(medians_path):
        medians = joblib.load(medians_path)
    else:
        medians = {}
        print("  ⚠️  feature_medians.pkl absent — imputation par 0 en fallback")

    # ── Date ─────────────────────────────────────────────────────────────────
    date = pd.Timestamp(date) if date is not None else pd.Timestamp.today()

    # ── Stats rolling ─────────────────────────────────────────────────────────
    home_stats = get_team_rolling_stats(df_historical, home_team, date, "home")
    away_stats = get_team_rolling_stats(df_historical, away_team, date, "away")

    # ── Classement ────────────────────────────────────────────────────────────
    home_rank_norm = get_rank_norm(df_historical, home_team, date)
    away_rank_norm = get_rank_norm(df_historical, away_team, date)
    classement_diff = away_rank_norm - home_rank_norm

    # ── H2H ──────────────────────────────────────────────────────────────────
    h2h_home_win_rate = get_h2h_win_rate(df_historical, home_team, away_team, date)

    # ── Différentiels de base ────────────────────────────────────────────────
    goals_diff   = home_stats["goals_scored_mean"]   - away_stats["goals_scored_mean"]
    defense_diff = away_stats["goals_conceded_mean"] - home_stats["goals_conceded_mean"]
    form_diff    = home_stats["form_sum"]            - away_stats["form_sum"]

    # ── Features dérivées (draw signal) ──────────────────────────────────────
    derived = compute_derived_features(
        goals_diff     = goals_diff,
        home_form      = home_stats["form_sum"],
        away_form      = away_stats["form_sum"],
        classement_diff= classement_diff,
        form_diff      = form_diff,
    )

    # ── Dictionnaire complet ─────────────────────────────────────────────────
    feature_dict = {
        "home_goals_avg":          home_stats["goals_scored_mean"],
        "away_goals_avg":          away_stats["goals_scored_mean"],
        "home_goals_conceded_avg": home_stats["goals_conceded_mean"],
        "away_goals_conceded_avg": away_stats["goals_conceded_mean"],
        "home_form":               home_stats["form_sum"],
        "away_form":               away_stats["form_sum"],
        "goals_diff":              goals_diff,
        "defense_diff":            defense_diff,
        "form_diff":               form_diff,
        "classement_diff":         classement_diff,
        "home_rank_norm":          home_rank_norm,
        "away_rank_norm":          away_rank_norm,
        "home_shots_target":       np.nan,   # indisponible avant le match
        "away_shots_target":       np.nan,
        "h2h_home_win_rate":       h2h_home_win_rate,
        "fatigue_diff":            away_stats["days_since_last"] - home_stats["days_since_last"],
        "strength_symmetry":       derived["strength_symmetry"],
        "draw_prior":              derived["draw_prior"],
        "low_stakes_proxy":        np.nan,   # non calculable sans contexte saison
        "home_attack_vs_away_def": home_stats["goals_scored_mean"] / max(away_stats["goals_conceded_mean"], 0.3),
        "away_attack_vs_home_def": away_stats["goals_scored_mean"] / max(home_stats["goals_conceded_mean"], 0.3),
    }

    # ── Vérification de cohérence ─────────────────────────────────────────────
    missing = [f for f in feature_cols if f not in feature_dict]
    if missing:
        raise KeyError(
            f"\n❌ Features présentes dans feature_names.txt mais absentes de predictor.py :\n"
            f"   {missing}\n"
            f"   → Ajoute-les dans feature_dict ci-dessus."
        )

    # ── Construction de X dans l'ordre exact de feature_names.txt ────────────
    X = np.array([[feature_dict[f] for f in feature_cols]], dtype=float)

    # Imputation : médianes de l'entraînement, puis 0 en dernier recours
    for i, col in enumerate(feature_cols):
        if np.isnan(X[0, i]):
            X[0, i] = float(medians.get(col, 0.0))

    # ── Prédiction ────────────────────────────────────────────────────────────
    probas = model.predict_proba(X)[0]   # [p_ext_gagne, p_nul, p_dom_gagne]

    # ── Score le plus probable via Poisson ────────────────────────────────────
    lambda_home = home_stats["goals_scored_mean"]
    lambda_away = away_stats["goals_scored_mean"]
    buts        = np.arange(0, 7)
    matrice     = np.outer(poisson.pmf(buts, lambda_home), poisson.pmf(buts, lambda_away))
    best_score  = np.unravel_index(matrice.argmax(), matrice.shape)

    result = {
        "match":               f"{home_team} vs {away_team}",
        "prob_home_win":       float(probas[2]),
        "prob_draw":           float(probas[1]),
        "prob_away_win":       float(probas[0]),
        "expected_goals_home": round(lambda_home, 2),
        "expected_goals_away": round(lambda_away, 2),
        "most_likely_score":   f"{best_score[0]} - {best_score[1]}",
    }

    _print_result(result, home_team, away_team)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Affichage
# ──────────────────────────────────────────────────────────────────────────────

def _print_result(result: dict, home_team: str, away_team: str) -> None:
    print(f"\n{'='*52}")
    print(f"  ⚽  Prédiction : {home_team} vs {away_team}")
    print(f"{'='*52}")

    items = [
        (f"Victoire {home_team}", result["prob_home_win"]),
        ("Match nul",             result["prob_draw"]),
        (f"Victoire {away_team}", result["prob_away_win"]),
    ]
    for label, prob in items:
        bar = "█" * int(prob * 30)
        print(f"  {label:<24} {prob:5.1%}  {bar}")

    print(f"\n  Score le plus probable : {result['most_likely_score']}")
    print(f"  xG dom / ext            : {result['expected_goals_home']} / {result['expected_goals_away']}")
    print(f"{'='*52}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédire l'issue d'un match de Ligue 1")
    parser.add_argument("--home",  required=True,  help="Équipe domicile (ex: PSG)")
    parser.add_argument("--away",  required=True,  help="Équipe extérieur (ex: Marseille)")
    parser.add_argument("--date",  default=None,   help="Date du match YYYY-MM-DD (défaut: aujourd'hui)")
    parser.add_argument("--data",  default="data/processed/ligue1_clean.csv",
                        help="Chemin vers le CSV historique")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"{args.data} introuvable. Lance d'abord : python src/data_loader.py"
        )

    df = pd.read_csv(args.data, parse_dates=["date"])
    predict_match(args.home, args.away, df, date=args.date)