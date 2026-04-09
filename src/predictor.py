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
import numpy as np
import pandas as pd
import joblib
from scipy.stats import poisson
from feature_engineering import get_team_stats_before
from model import FEATURE_COLS

def predict_match(home_team, away_team, df_historical, date=None):
    """
    Prédit les probabilités d'issue d'un match.

    Args:
        home_team (str): Nom de l'équipe à domicile (doit correspondre aux données)
        away_team (str): Nom de l'équipe à l'extérieur
        df_historical (pd.DataFrame): Historique des matchs
        date (str ou pd.Timestamp, optional): Date du match. Défaut : aujourd'hui.

    Returns:
        dict: {
            'match': str,
            'prob_home_win': float,
            'prob_draw': float,
            'prob_away_win': float,
            'expected_goals_home': float,
            'expected_goals_away': float,
            'most_likely_score': str,
        }
    """
    model_path = "models/xgb_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} introuvable. Lance d'abord : python src/model.py"
        )

    if date is None:
        date = pd.Timestamp.today()
    else:
        date = pd.Timestamp(date)

    model = joblib.load(model_path)

    # Calcul des features pour ce match
    home_stats = get_team_stats_before(df_historical, home_team, date, "home")
    away_stats = get_team_stats_before(df_historical, away_team, date, "away")
    all_home   = get_team_stats_before(df_historical, home_team, date, "both")
    all_away   = get_team_stats_before(df_historical, away_team, date, "both")

    standings = {}
    try:
        # Reconstruire le classement depuis les données historiques
        from feature_engineering import compute_standings
        standings = compute_standings(df_historical, date)
    except Exception:
        pass

    n_teams        = len(standings) if standings else 20
    home_rank      = standings.get(home_team, 10)
    away_rank      = standings.get(away_team, 10)
    home_rank_norm = (home_rank - 1) / max(n_teams - 1, 1)
    away_rank_norm = (away_rank - 1) / max(n_teams - 1, 1)

    X = np.array([[
        home_stats['goals_scored_mean'],
        away_stats['goals_scored_mean'],
        home_stats['goals_conceded_mean'],
        away_stats['goals_conceded_mean'],
        all_home['points_sum'],
        all_away['points_sum'],
        home_stats['goals_scored_mean']   - away_stats['goals_scored_mean'],
        away_stats['goals_conceded_mean'] - home_stats['goals_conceded_mean'],
        all_home['points_sum']            - all_away['points_sum'],
        # Nouvelles features
        away_rank_norm - home_rank_norm,
        home_rank_norm,
        away_rank_norm,
        home_stats['goals_scored_mean'] / max(away_stats['goals_conceded_mean'], 0.3),
        away_stats['goals_scored_mean'] / max(home_stats['goals_conceded_mean'], 0.3),
        np.nan,  # home_shots_target
        np.nan,  # away_shots_target
    ]])

    # Remplacer les NaN par la médiane (cohérent avec l'entraînement)
    X = np.nan_to_num(X, nan=1.5)

    # Prédiction ML
    probas = model.predict_proba(X)[0]  # [p_ext_gagne, p_nul, p_dom_gagne]

    # Score le plus probable via Poisson
    lambda_home = home_stats["goals_scored_mean"]
    lambda_away = away_stats["goals_scored_mean"]
    buts = np.arange(0, 7)
    matrice = np.outer(poisson.pmf(buts, lambda_home), poisson.pmf(buts, lambda_away))
    best_score = np.unravel_index(matrice.argmax(), matrice.shape)

    result = {
        "match":               f"{home_team} vs {away_team}",
        "prob_home_win":       float(probas[2]),
        "prob_draw":           float(probas[1]),
        "prob_away_win":       float(probas[0]),
        "expected_goals_home": round(lambda_home, 2),
        "expected_goals_away": round(lambda_away, 2),
        "most_likely_score":   f"{best_score[0]} - {best_score[1]}",
    }

    # Affichage formaté
    _print_result(result, home_team, away_team)

    return result


def _print_result(result, home_team, away_team):
    """Affiche les résultats de prédiction de façon lisible."""
    print(f"\n{'='*50}")
    print(f"  ⚽  Prédiction : {home_team} vs {away_team}")
    print(f"{'='*50}")

    items = [
        (f"Victoire {home_team}", result["prob_home_win"]),
        ("Match nul",             result["prob_draw"]),
        (f"Victoire {away_team}", result["prob_away_win"]),
    ]
    for label, prob in items:
        bar = "█" * int(prob * 30)
        print(f"  {label:<22} {prob:.1%}  {bar}")

    print(f"\n  Score le plus probable : {result['most_likely_score']}")
    print(f"  xG domicile / extérieur : {result['expected_goals_home']} / {result['expected_goals_away']}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédire l'issue d'un match de football")
    parser.add_argument("--home",  required=True,  help="Équipe à domicile (ex: PSG)")
    parser.add_argument("--away",  required=True,  help="Équipe à l'extérieur (ex: Lyon)")
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
