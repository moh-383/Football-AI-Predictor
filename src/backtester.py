<<<<<<< HEAD
"""
backtester.py
-------------
Backtesting en conditions "aveugles" sur les N derniers matchs du dataset.

Le backtesting simule une utilisation réelle :
  - Pour chaque match du test set, le modèle ne voit QUE les données antérieures.
  - Aucune information future ne filtre (zero leakage).

Utilisation :
    python src/backtester.py [--n 200] [--report]

Entrée  : data/processed/features.csv, models/xgb_model.pkl
Sortie  : rapport console + optionnel models/backtest_report.csv
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

# Import depuis model.py (même répertoire)
import sys
sys.path.insert(0, os.path.dirname(__file__))
from model import FEATURE_COLS, add_draw_features, XGB_PARAMS, USE_CLASS_WEIGHTS


# ──────────────────────────────────────────────────────────────────────────────
# Chargement des artefacts
# ──────────────────────────────────────────────────────────────────────────────

def load_artifacts(model_dir: str = "models") -> tuple:
    """
    Charge le modèle XGBoost et les médianes d'imputation.

    Returns:
        (model, medians) — medians est un pd.Series ou None
    """
    model_path   = os.path.join(model_dir, "xgb_model.pkl")
    medians_path = os.path.join(model_dir, "feature_medians.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} introuvable. Lance d'abord : python src/model.py"
        )

    model   = joblib.load(model_path)
    medians = joblib.load(medians_path) if os.path.exists(medians_path) else None

    return model, medians


def prepare_X(test_df: pd.DataFrame, medians: pd.Series | None) -> np.ndarray:
    """
    Prépare la matrice de features avec imputation cohérente.
    Applique add_draw_features (constantes fixes = zéro leakage).
    Utilise les médianes d'entraînement pour l'imputation.
    """
    df = add_draw_features(test_df)
    X_df = df.reindex(columns=FEATURE_COLS)

    if medians is not None:
        X_df = X_df.fillna(medians)
    else:
        X_df = X_df.fillna(X_df.median())

    return X_df.values


# ──────────────────────────────────────────────────────────────────────────────
# Backtesting principal
# ──────────────────────────────────────────────────────────────────────────────

def backtest(
    features_df: pd.DataFrame,
    model,
    medians: pd.Series | None,
    n_test_matches: int = 200,
) -> pd.DataFrame:
    """
    Prédit les N derniers matchs du dataset (dont on connaît le résultat)
    et calcule les métriques de précision réelle.

    Args:
        features_df:    DataFrame de features (output feature_engineering.py)
        model:          Modèle XGBoost chargé
        medians:        Médianes d'imputation (du training set)
        n_test_matches: Nombre de matchs de test (les plus récents)

    Returns:
        pd.DataFrame des prédictions avec colonnes supplémentaires
    """
    # Tri chronologique strict (critique)
    features_df = features_df.sort_values("date").reset_index(drop=True)

    train_set = features_df.iloc[:-n_test_matches]
    test_set  = features_df.iloc[-n_test_matches:].copy()

    print(f"  Train : {len(train_set)} matchs ({train_set['date'].min().date()} → {train_set['date'].max().date()})")
    print(f"  Test  : {len(test_set)} matchs  ({test_set['date'].min().date()} → {test_set['date'].max().date()})\n")

    # ── ANTI-LEAKAGE : entraîner un modèle dédié sur train_set uniquement ──────
    # Le modèle chargé depuis models/xgb_model.pkl est entraîné sur 100%
    # des données (incluant le test set) → l'utiliser pour évaluer le test
    # set revient à mesurer la performance en mémoire, pas en généralisation.
    # On entraîne ici un modèle identique mais limité aux 1525 matchs passés.
    print("  Entraînement du modèle de backtesting (train set uniquement)…")
    train_feats = add_draw_features(train_set)
    X_train_bt  = train_feats.reindex(columns=FEATURE_COLS)
    medians_bt  = X_train_bt.median()
    X_train_bt  = X_train_bt.fillna(medians_bt).values
    y_train_bt  = train_set["target"].astype(int).values

    if USE_CLASS_WEIGHTS:
        classes = np.array([0, 1, 2])
        cw = compute_class_weight("balanced", classes=classes, y=y_train_bt)
        sw = np.array([cw[yi] for yi in y_train_bt])
    else:
        sw = None

    bt_model = XGBClassifier(**XGB_PARAMS)
    bt_model.fit(X_train_bt, y_train_bt, sample_weight=sw, verbose=False)
    print("  Modèle de backtesting prêt.")

    X_test = prepare_X(test_set, medians_bt)
    y_true = test_set["target"].astype(int).values

    y_pred  = bt_model.predict(X_test)
    y_proba = bt_model.predict_proba(X_test)   # shape (n, 3) : [ext, nul, dom]

    # ── Métriques globales ───────────────────────────────────────────────────
    acc      = accuracy_score(y_true, y_pred)
    loss     = log_loss(y_true, y_proba)
    baseline = (y_true == 2).mean()

    print(f"  {'─'*50}")
    print(f"  Accuracy réelle     : {acc:.3f}  (baseline naïve : {baseline:.3f})")
    print(f"  Gain vs baseline    : {acc - baseline:+.3f}")
    print(f"  Log Loss            : {loss:.4f}")
    print(f"  {'─'*50}\n")

    # ── Classification report ────────────────────────────────────────────────
    label_map = {0: "Ext gagne", 1: "Nul", 2: "Dom gagne"}
    target_names = [label_map[i] for i in sorted(set(y_true))]
    print(classification_report(y_true, y_pred, target_names=target_names))

    # ── Matrice de confusion ─────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    print("  Matrice de confusion (lignes = réel, colonnes = prédit) :")
    print(f"  {'':12} {'Ext':>8} {'Nul':>8} {'Dom':>8}")
    for i, row_label in enumerate(["Ext gagne", "Nul      ", "Dom gagne"]):
        print(f"  {row_label}  " + "  ".join(f"{cm[i,j]:>8}" for j in range(3)))

    # ── Matchs où le modèle s'est le plus trompé ─────────────────────────────
    test_out = test_set.copy()
    test_out["y_true"]      = y_true
    test_out["y_pred"]      = y_pred
    test_out["correct"]     = (y_pred == y_true)
    test_out["prob_dom"]    = y_proba[:, 2]
    test_out["prob_nul"]    = y_proba[:, 1]
    test_out["prob_ext"]    = y_proba[:, 0]
    test_out["confidence"]  = y_proba.max(axis=1)
    test_out["error"]       = ~test_out["correct"]

    # Top 10 erreurs à forte confiance (les plus révélatrices)
    confident_errors = (
        test_out[test_out["error"]]
        .sort_values("confidence", ascending=False)
        .head(10)
    )

    if len(confident_errors) > 0:
        print("\n  ⚠️  Top 10 erreurs à forte confiance :")
        print(f"  {'Match':<35} {'Réel':>10} {'Prédit':>10} {'Confiance':>10}")
        print(f"  {'─'*68}")
        for _, r in confident_errors.iterrows():
            match_str = f"{r.get('home_team','?')} vs {r.get('away_team','?')}"
            print(
                f"  {match_str:<35}"
                f" {label_map.get(int(r['y_true']), '?'):>10}"
                f" {label_map.get(int(r['y_pred']), '?'):>10}"
                f" {r['confidence']:>9.1%}"
            )

    return test_out


# ──────────────────────────────────────────────────────────────────────────────
# Analyse des matchs difficiles à prédire
# ──────────────────────────────────────────────────────────────────────────────

def analyze_hard_matches(test_out: pd.DataFrame) -> None:
    """
    Identifie les profils de matchs difficiles à prédire.
    Analyse la relation entre confiance du modèle et précision réelle.
    """
    print("\n  Calibration : précision par niveau de confiance\n")
    bins   = [0.33, 0.40, 0.50, 0.60, 0.70, 1.01]
    labels = ["33-40%", "40-50%", "50-60%", "60-70%", "70%+"]

    test_out["conf_bin"] = pd.cut(test_out["confidence"], bins=bins, labels=labels)

    cal = (
        test_out.groupby("conf_bin", observed=True)["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
    )

    for label, row in cal.iterrows():
        bar = "█" * int(row["accuracy"] * 30)
        print(f"  Confiance {label}  | Acc {row['accuracy']:.1%}  {bar}  (n={int(row['n'])})")


# ──────────────────────────────────────────────────────────────────────────────
# Entrée principale
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtesting Football AI Predictor")
    parser.add_argument("--n",      type=int, default=200, help="Nombre de matchs de test")
    parser.add_argument("--report", action="store_true",   help="Sauvegarder le rapport CSV")
    args = parser.parse_args()

    print(f"\n=== Backtesting sur {args.n} matchs récents ===\n")

    features_path = "data/processed/features.csv"
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"{features_path} introuvable. Lance d'abord : python src/feature_engineering.py"
        )

    features_df = pd.read_csv(features_path, parse_dates=["date"])
    features_df = features_df.dropna(subset=["target"])

    model, medians = load_artifacts()

    test_out = backtest(features_df, model, medians, n_test_matches=args.n)
    analyze_hard_matches(test_out)

    if args.report:
        report_path = "models/backtest_report.csv"
        test_out.to_csv(report_path, index=False)
        print(f"\n  ✅ Rapport sauvegardé : {report_path}")

    print()
=======
def backtest(df, model, features_df, n_test_matches=200):
    """
    Predit les N derniers matchs du dataset (dont on connait le resultat)
    et calcule la precision reelle du modele.
    """
    test_set = features_df.tail(n_test_matches)
    X_test = test_set[FEATURE_COLS].values
    y_true = test_set["target"].values
    y_pred = model.predict(X_test)
 
    accuracy = (y_pred == y_true).mean()
    print(f"Accuracy sur {n_test_matches} matchs recents : {accuracy:.1%}")
 
    # Afficher les matchs ou le modele s'est le plus trompe
    # -> identifier les types de matchs difficiles a predire
>>>>>>> 4aa703652f267a9f622bad050fa833cfaccb0b8e
