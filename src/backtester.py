"""
backtester.py — Football AI Predictor v3.0
===========================================
Backtesting en conditions "aveugles" sur les N derniers matchs.
Zéro leakage : le modèle de backtest est entraîné uniquement sur les données antérieures.

Usage :
    python src/backtester.py [--n 200] [--report] [--shap]
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from model import (
    FEATURE_COLS,
    XGB_PARAMS,
    CLASS_WEIGHTS,
    add_draw_features,
    prepare_X_y,
    _make_sample_weights,
    run_shap_analysis,
)


LABEL_MAP = {0: "Ext gagne", 1: "Nul      ", 2: "Dom gagne"}
LABEL_SHORT = {0: "Ext", 1: "Nul", 2: "Dom"}


# ──────────────────────────────────────────────────────────────────────────────
# Chargement des artefacts de production
# ──────────────────────────────────────────────────────────────────────────────

def load_artifacts(model_dir: str = "models") -> tuple:
    model_path   = os.path.join(model_dir, "xgb_model.pkl")
    medians_path = os.path.join(model_dir, "feature_medians.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} → lance python src/model.py")

    model   = joblib.load(model_path)
    medians = joblib.load(medians_path) if os.path.exists(medians_path) else None
    return model, medians


def prepare_X_bt(test_df: pd.DataFrame, medians: pd.Series | None) -> np.ndarray:
    """Prépare la matrice de features avec imputation cohérente."""
    df   = add_draw_features(test_df)
    X_df = df.reindex(columns=FEATURE_COLS)
    X_df = X_df.fillna(medians) if medians is not None else X_df.fillna(X_df.median())
    return X_df.values


# ──────────────────────────────────────────────────────────────────────────────
# Backtesting principal
# ──────────────────────────────────────────────────────────────────────────────

def backtest(
    features_df: pd.DataFrame,
    n_test_matches: int = 200,
    run_shap: bool = False,
) -> pd.DataFrame:
    """
    Prédit les N derniers matchs et calcule les métriques de précision réelle.

    Le modèle de backtest est entraîné UNIQUEMENT sur les données antérieures
    (train_set) pour garantir un zéro leakage absolu.
    """
    features_df = features_df.sort_values("date").reset_index(drop=True)
    train_set   = features_df.iloc[:-n_test_matches]
    test_set    = features_df.iloc[-n_test_matches:].copy()

    print(f"  Train : {len(train_set):4d} matchs "
          f"({train_set['date'].min().date()} → {train_set['date'].max().date()})")
    print(f"  Test  : {len(test_set):4d} matchs "
          f"({test_set['date'].min().date()} → {test_set['date'].max().date()})\n")

    # ── Entraînement du modèle de backtest (train only) ───────────────────────
    print("  Entraînement modèle backtest (train uniquement)…")
    train_feats  = add_draw_features(train_set)
    X_train_bt   = train_feats.reindex(columns=FEATURE_COLS)
    medians_bt   = X_train_bt.median()
    X_train_bt   = X_train_bt.fillna(medians_bt).values
    y_train_bt   = train_set["target"].astype(int).values

    sw = _make_sample_weights(y_train_bt)

    bt_model = XGBClassifier(**XGB_PARAMS)
    bt_model.fit(X_train_bt, y_train_bt, sample_weight=sw, verbose=False)
    print("  Modèle de backtest prêt.\n")

    X_test = prepare_X_bt(test_set, medians_bt)
    y_true = test_set["target"].astype(int).values

    y_pred  = bt_model.predict(X_test)
    y_proba = bt_model.predict_proba(X_test)   # (n, 3) : [ext, nul, dom]

    # ── Métriques globales ────────────────────────────────────────────────────
    acc      = accuracy_score(y_true, y_pred)
    loss     = log_loss(y_true, y_proba)
    baseline = (y_true == 2).mean()

    print(f"  {'─'*52}")
    print(f"  Accuracy réelle    : {acc:.3f}  (baseline naïve : {baseline:.3f})")
    print(f"  Gain vs baseline   : {acc - baseline:+.4f}")
    print(f"  Log Loss           : {loss:.4f}")
    print(f"  {'─'*52}\n")

    # ── Distribution des prédictions ──────────────────────────────────────────
    dist = {k: (y_pred == k).mean() for k in [0, 1, 2]}
    print("  Distribution prédictions :")
    for k, v in dist.items():
        print(f"    {LABEL_MAP[k]} : {v:.1%}")
    print()

    # ── Classification report ─────────────────────────────────────────────────
    target_names = ["Ext gagne", "Nul", "Dom gagne"]
    print(classification_report(y_true, y_pred, target_names=target_names))

    # ── Matrice de confusion ──────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    _print_confusion_matrix(cm)

    # ── Enrichissement du DataFrame test ─────────────────────────────────────
    test_out              = test_set.copy()
    test_out["y_true"]    = y_true
    test_out["y_pred"]    = y_pred
    test_out["correct"]   = (y_pred == y_true)
    test_out["prob_dom"]  = y_proba[:, 2]
    test_out["prob_nul"]  = y_proba[:, 1]
    test_out["prob_ext"]  = y_proba[:, 0]
    test_out["confidence"]= y_proba.max(axis=1)
    test_out["error"]     = ~test_out["correct"]

    # ── Top erreurs à forte confiance ─────────────────────────────────────────
    _print_top_errors(test_out, n=10)

    # ── SHAP optionnel ────────────────────────────────────────────────────────
    if run_shap:
        run_shap_analysis(bt_model, X_test, output_dir="models")

    return test_out


# ──────────────────────────────────────────────────────────────────────────────
# Affichage
# ──────────────────────────────────────────────────────────────────────────────

def _print_confusion_matrix(cm: np.ndarray) -> None:
    labels = ["Ext gagne", "Nul      ", "Dom gagne"]
    print("  Matrice de confusion (lignes = réel, colonnes = prédit) :")
    print(f"  {'':14} {'Ext':>8} {'Nul':>8} {'Dom':>8}")
    for i, lbl in enumerate(labels):
        row = "  ".join(f"{cm[i, j]:>8}" for j in range(3))
        print(f"  {lbl}  {row}")
    print()


def _print_top_errors(test_out: pd.DataFrame, n: int = 10) -> None:
    errors = (
        test_out[test_out["error"]]
        .sort_values("confidence", ascending=False)
        .head(n)
    )
    if len(errors) == 0:
        return

    print(f"\n  ⚠️  Top {n} erreurs à forte confiance :")
    print(f"  {'Match':<36} {'Réel':>10} {'Prédit':>10} {'Conf':>9}")
    print(f"  {'─'*70}")
    for _, r in errors.iterrows():
        match_str = f"{r.get('home_team','?')} vs {r.get('away_team','?')}"
        true_lbl  = LABEL_MAP.get(int(r["y_true"]), "?").strip()
        pred_lbl  = LABEL_MAP.get(int(r["y_pred"]), "?").strip()
        print(
            f"  {match_str:<36}"
            f" {true_lbl:>10}"
            f" {pred_lbl:>10}"
            f" {r['confidence']:>8.1%}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Calibration : précision réelle par niveau de confiance
# ──────────────────────────────────────────────────────────────────────────────

def analyze_calibration(test_out: pd.DataFrame) -> None:
    """
    Identifie la zone morte de calibration (typiquement 50-60% en v2.x).
    En v3.0, cette zone doit être ≥ 45% grâce aux poids de classe manuels.
    """
    print("\n  Calibration : précision réelle par niveau de confiance\n")
    bins   = [0.33, 0.40, 0.50, 0.60, 0.70, 1.01]
    labels = ["33-40%", "40-50%", "50-60%", "60-70%", "70%+"]

    test_out = test_out.copy()
    test_out["conf_bin"] = pd.cut(test_out["confidence"], bins=bins, labels=labels)

    cal = (
        test_out.groupby("conf_bin", observed=True)["correct"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "accuracy", "count": "n"})
    )

    for label, row in cal.iterrows():
        bar   = "█" * int(row["accuracy"] * 32)
        alert = " ← ⚠️  ZONE MORTE" if row["accuracy"] < 0.40 else ""
        print(
            f"  {label:>8}  | Acc {row['accuracy']:.1%}  {bar}{alert}"
            f"  (n={int(row['n'])})"
        )

    print()
    # Analyse spéciale zone 50-60% (zone morte v2.x)
    zone = test_out[test_out["conf_bin"] == "50-60%"]
    if len(zone) > 0:
        z_acc = zone["correct"].mean()
        z_nul = (zone["y_true"] == 1).mean()
        print(f"  Analyse zone 50-60% :")
        print(f"    Accuracy : {z_acc:.1%}  |  Taux nul réel : {z_nul:.1%}")
        if z_acc >= 0.45:
            print("    ✅ Zone morte corrigée (v3.0)")
        else:
            print("    ⚠️  Zone morte persistante — ajuster CLASS_WEIGHTS[1]")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtesting Football AI Predictor v3.0")
    parser.add_argument("--n",      type=int, default=200, help="Matchs de test (les plus récents)")
    parser.add_argument("--report", action="store_true",   help="Sauvegarder CSV")
    parser.add_argument("--shap",   action="store_true",   help="SHAP sur le modèle de backtest")
    args = parser.parse_args()

    print(f"\n=== Backtesting v3.0 — {args.n} matchs récents ===\n")

    path = "data/processed/features.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} → lance feature_engineering.py")

    features_df = pd.read_csv(path, parse_dates=["date"])
    features_df = features_df.dropna(subset=["target"])

    test_out = backtest(features_df, n_test_matches=args.n, run_shap=args.shap)
    analyze_calibration(test_out)

    if args.report:
        out = "models/backtest_report.csv"
        test_out.to_csv(out, index=False)
        print(f"  ✅ Rapport : {out}")

    print()