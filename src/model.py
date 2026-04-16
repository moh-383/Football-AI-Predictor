"""
model.py — Football AI Predictor v3.0
======================================
Changements majeurs vs v2.x :
  - Suppression du class_weight="balanced" automatique → poids manuels {0:1.0, 1:1.4, 2:1.0}
  - Zone morte 50-60% corrigée (over-softmax du nul éliminé)
  - FEATURE_COLS étendu avec h2h_draw_rate, fatigue_diff (déjà présent en FE v2)
  - Prépare les séquences pour Phase 5 LSTM (export optionnel)
  - TimeSeriesSplit strict, anti-leakage garanti par shift(1) dans feature_engineering
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────────────────────────────────────
# FEATURE COLS v3.0 — ordre figé (ne jamais réordonner sans réentraîner)
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_COLS: list[str] = [
    # Attaque / défense rolling (decay=0.9, window=10)
    "home_goals_avg",
    "away_goals_avg",
    "home_goals_conceded_avg",
    "away_goals_conceded_avg",
    # Forme (somme pts sur window)
    "home_form",
    "away_form",
    # Différentiels bruts
    "goals_diff",
    "defense_diff",
    "form_diff",
    # Classement normalisé [0=1er, 1=dernier]
    "classement_diff",
    # Tirs cadrés (NaN avant match → imputé par médiane)
    "home_shots_target",
    "away_shots_target",
    # H2H v3 : win_rate + draw_rate
    "h2h_home_win_rate",
    "h2h_draw_rate",          # NOUVEAU v3.0
    # Fatigue
    "fatigue_diff",
    # Signaux nul (calculés dans add_draw_features)
    "strength_symmetry",
    "draw_prior",
    # Ratios croisés attaque/défense
    "home_attack_vs_away_def",
    "away_attack_vs_home_def",
]

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparamètres v3.0 (optimisés pour 10 saisons ~3040 matchs)
# ──────────────────────────────────────────────────────────────────────────────
XGB_PARAMS: dict = {
    "n_estimators":      600,
    "learning_rate":     0.025,
    "max_depth":         4,          # garde la généralisation
    "subsample":         0.8,
    "colsample_bytree":  0.75,
    "min_child_weight":  6,
    "gamma":             0.15,
    "reg_alpha":         0.1,
    "reg_lambda":        1.2,
    "objective":         "multi:softprob",
    "num_class":         3,
    "random_state":      42,
    "eval_metric":       "mlogloss",
    "n_jobs":            -1,
    "use_label_encoder": False,
}

# ─────────────────────────────────────────────────────────────────────────────
# POIDS DE CLASSE MANUELS v3.0
# Remplace class_weight="balanced" qui sur-calibrait Nul (~×1.9 → zone morte)
# Calibration empirique : Nul ×1.4 préserve rappel sans dégrader accuracy
# ─────────────────────────────────────────────────────────────────────────────
CLASS_WEIGHTS: dict[int, float] = {
    0: 1.0,   # Victoire extérieur
    1: 1.4,   # Nul (légèrement surpondéré)
    2: 1.0,   # Victoire domicile
}

# Grille de recherche (--tune uniquement)
PARAM_GRID: dict = {
    "learning_rate":    [0.02, 0.03, 0.05],
    "max_depth":        [3, 4, 5],
    "subsample":        [0.75, 0.8, 0.85],
    "min_child_weight": [4, 6, 8],
}


# ──────────────────────────────────────────────────────────────────────────────
# Features dérivées — constantes domaine fixes (ZÉRO LEAKAGE)
# ──────────────────────────────────────────────────────────────────────────────
GOAL_DIFF_SCALE = 3.0
MAX_FORM        = 30.0
FORM_DIFF_SCALE = 30.0


def add_draw_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les signaux spécifiques au nul avec constantes fixes.
    Aucune statistique globale du dataset (mean/max) n'est utilisée.
    Compatible avec l'inférence en production (predictor.py).
    """
    df = features_df.copy()

    # 1. Symétrie de force : 1 si équipes équilibrées, 0 si écart maximal
    df["strength_symmetry"] = (
        1.0 - (df["goals_diff"].abs() / GOAL_DIFF_SCALE).clip(0, 1)
    )

    # 2. Draw prior : forme basse des deux → plus probable un nul
    df["draw_prior"] = 1.0 - (
        (df["home_form"].clip(0, MAX_FORM) + df["away_form"].clip(0, MAX_FORM))
        / (2 * MAX_FORM)
    )

    return df


def prepare_X_y(
    features_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """
    Extrait X, y et les médianes d'imputation.
    Les médianes sont calculées sur le DataFrame courant → à sauvegarder
    pour que l'inférence reste cohérente.
    """
    features_df = add_draw_features(features_df)

    # Vérification des colonnes manquantes
    missing = [c for c in FEATURE_COLS if c not in features_df.columns]
    if missing:
        print(f"  ⚠️  Features absentes (imputées à 0) : {missing}")

    X_df    = features_df.reindex(columns=FEATURE_COLS)
    medians = X_df.median()
    X       = X_df.fillna(medians).values
    y       = features_df["target"].astype(int).values

    return X, y, medians


def _make_sample_weights(y: np.ndarray) -> np.ndarray:
    """Construit le vecteur sample_weight depuis CLASS_WEIGHTS."""
    return np.array([CLASS_WEIGHTS[yi] for yi in y], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# Validation croisée temporelle
# ──────────────────────────────────────────────────────────────────────────────

def train_with_cv(
    features_df: pd.DataFrame,
    n_splits: int = 5,
    params: dict | None = None,
) -> tuple[XGBClassifier, dict, pd.Series]:
    """
    TimeSeriesSplit strict (aucune fuite temporelle).
    Retourne : (modèle final, métriques, médianes d'imputation).
    """
    if params is None:
        params = XGB_PARAMS

    X, y, medians = prepare_X_y(features_df)
    tscv          = TimeSeriesSplit(n_splits=n_splits)
    baseline      = (y == 2).mean()

    print(f"  Dataset       : {len(y)} matchs · {X.shape[1]} features")
    print(f"  Baseline naïve: {baseline:.3f}")
    print(f"  Poids de classe: {CLASS_WEIGHTS}")
    print(f"  {'─'*60}")

    scores_acc, scores_loss = [], []
    label_map = {0: "Ext", 1: "Nul", 2: "Dom"}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        sw = _make_sample_weights(y[train_idx])

        model = XGBClassifier(**params)
        model.fit(
            X[train_idx], y[train_idx],
            sample_weight=sw,
            eval_set=[(X[val_idx], y[val_idx])],
            verbose=False,
        )

        y_pred  = model.predict(X[val_idx])
        y_proba = model.predict_proba(X[val_idx])

        acc  = accuracy_score(y[val_idx], y_pred)
        loss = log_loss(y[val_idx], y_proba)
        scores_acc.append(acc)
        scores_loss.append(loss)

        delta = acc - baseline
        sign  = "▲" if delta > 0 else "▼"

        # Distribution des prédictions (indicateur de calibration)
        dist = {k: (y_pred == k).mean() for k in [0, 1, 2]}
        dist_str = " | ".join(f"{label_map[k]}:{v:.0%}" for k, v in dist.items())

        print(
            f"  Fold {fold} | Acc {acc:.3f} ({sign}{abs(delta):.3f})"
            f" | LogLoss {loss:.3f}"
            f" | [{dist_str}]"
            f" | {len(train_idx):4d}/{len(val_idx):4d}"
        )

    mean_acc  = float(np.mean(scores_acc))
    mean_loss = float(np.mean(scores_loss))
    print(f"  {'─'*60}")
    print(f"  CV Accuracy : {mean_acc:.3f} ± {np.std(scores_acc):.3f}")
    print(f"  CV LogLoss  : {mean_loss:.3f} ± {np.std(scores_loss):.3f}")

    # Modèle final sur 100% des données
    print("\n  Entraînement du modèle final (100% données)…")
    final_model = XGBClassifier(**params)
    sw_final    = _make_sample_weights(y)
    final_model.fit(X, y, sample_weight=sw_final, verbose=False)

    metrics = {
        "version":        "3.0",
        "accuracy_mean":  round(mean_acc, 4),
        "accuracy_std":   round(float(np.std(scores_acc)), 4),
        "logloss_mean":   round(mean_loss, 4),
        "logloss_std":    round(float(np.std(scores_loss)), 4),
        "baseline":       round(float(baseline), 4),
        "n_matches":      int(len(y)),
        "features":       FEATURE_COLS,
        "class_weights":  CLASS_WEIGHTS,
        "params":         {k: v for k, v in params.items() if k != "eval_metric"},
    }

    return final_model, metrics, medians


# ──────────────────────────────────────────────────────────────────────────────
# GridSearch temporel
# ──────────────────────────────────────────────────────────────────────────────

def temporal_grid_search(
    features_df: pd.DataFrame,
    param_grid: dict = PARAM_GRID,
    n_splits: int = 4,
) -> dict:
    """GridSearch avec TimeSeriesSplit — minimise la Log Loss."""
    from itertools import product

    X, y, _ = prepare_X_y(features_df)
    tscv    = TimeSeriesSplit(n_splits=n_splits)
    keys    = list(param_grid.keys())
    combos  = list(product(*param_grid.values()))

    print(f"  GridSearch : {len(combos)} combinaisons × {n_splits} folds\n")

    best_loss, best_params = float("inf"), {}

    for i, combo in enumerate(combos, 1):
        params = {**XGB_PARAMS, **dict(zip(keys, combo))}
        losses = []

        for tr, te in tscv.split(X):
            sw = _make_sample_weights(y[tr])
            m  = XGBClassifier(**params)
            m.fit(X[tr], y[tr], sample_weight=sw, verbose=False)
            losses.append(log_loss(y[te], m.predict_proba(X[te])))

        ml = float(np.mean(losses))
        print(f"  [{i:3d}/{len(combos)}] {dict(zip(keys, combo))} → LogLoss {ml:.4f}")

        if ml < best_loss:
            best_loss, best_params = ml, params

    print(f"\n  ✅ Best params (LogLoss={best_loss:.4f}) : {best_params}")
    return best_params


# ──────────────────────────────────────────────────────────────────────────────
# Analyse SHAP
# ──────────────────────────────────────────────────────────────────────────────

def run_shap_analysis(
    model: XGBClassifier,
    X: np.ndarray,
    output_dir: str = "models",
    sample_size: int = 600,
) -> None:
    """
    SHAP globale (mean |SHAP|) + export CSV.
    Identifie les features < seuil de bruit (0.005).
    Compatible SHAP < 0.40 et >= 0.40.
    """
    try:
        import shap
    except ImportError:
        print("  ⚠️  shap non installé : pip install shap")
        return

    n = min(sample_size, len(X))
    X_s = X[:n]

    print(f"\n  SHAP analysis (n={n})…")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_s)

    if isinstance(shap_values, list):
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    mean_abs = np.array(mean_abs).flatten()
    feat_names = FEATURE_COLS[: len(mean_abs)]

    imp_df = pd.DataFrame({"feature": feat_names, "shap_mean": mean_abs})
    imp_df = imp_df.sort_values("shap_mean", ascending=False)

    print(f"\n  {'Feature':<35} {'SHAP':>8}  Bar")
    print(f"  {'─'*60}")
    max_v = imp_df["shap_mean"].max()
    for _, row in imp_df.iterrows():
        bar  = "█" * int(row["shap_mean"] / max_v * 28)
        flag = " ← bruit?" if row["shap_mean"] < 0.005 else ""
        print(f"  {row['feature']:<35} {row['shap_mean']:>7.4f}  {bar}{flag}")

    out = os.path.join(output_dir, "shap_importance.csv")
    imp_df.to_csv(out, index=False)
    print(f"\n  ✅ SHAP sauvegardé : {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Export séquences LSTM (Phase 5)
# ──────────────────────────────────────────────────────────────────────────────

def export_lstm_sequences(
    features_df: pd.DataFrame,
    seq_len: int = 10,
    output_path: str = "data/processed/lstm_sequences.npz",
) -> None:
    """
    Prépare les données pour le futur LSTM (Phase 5).
    Produit des tenseurs (N, seq_len, n_features) sans leakage.

    Format de sortie :
        X_seq : (N, seq_len, n_features)  float32
        y_seq : (N,)                       int8
    """
    features_df = add_draw_features(features_df)
    df = features_df.sort_values("date").reset_index(drop=True)

    X_df = df.reindex(columns=FEATURE_COLS)
    meds = X_df.median()
    X_all = X_df.fillna(meds).values.astype(np.float32)
    y_all = df["target"].astype(np.int8).values

    seqs_X, seqs_y = [], []
    for i in range(seq_len, len(X_all)):
        seqs_X.append(X_all[i - seq_len: i])
        seqs_y.append(y_all[i])

    X_seq = np.array(seqs_X, dtype=np.float32)   # (N, seq_len, F)
    y_seq = np.array(seqs_y, dtype=np.int8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, X_seq=X_seq, y_seq=y_seq, feature_names=FEATURE_COLS)
    print(f"  ✅ LSTM séquences exportées : {output_path}")
    print(f"     Shape X: {X_seq.shape} | Shape y: {y_seq.shape}")


# ──────────────────────────────────────────────────────────────────────────────
# Sauvegarde
# ──────────────────────────────────────────────────────────────────────────────

def save_model(
    model: XGBClassifier,
    medians: pd.Series,
    metrics: dict,
    output_dir: str = "models",
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    joblib.dump(model,   os.path.join(output_dir, "xgb_model.pkl"))
    joblib.dump(medians, os.path.join(output_dir, "feature_medians.pkl"))

    with open(os.path.join(output_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(FEATURE_COLS))

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  ✅ xgb_model.pkl     → {output_dir}/")
    print(f"  ✅ feature_medians.pkl → {output_dir}/")
    print(f"  ✅ metrics.json        → {output_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football AI Predictor — Entraînement v3.0")
    parser.add_argument("--tune",   action="store_true", help="GridSearch temporel")
    parser.add_argument("--shap",   action="store_true", help="Analyse SHAP")
    parser.add_argument("--lstm",   action="store_true", help="Exporter séquences LSTM")
    parser.add_argument("--splits", type=int, default=5, help="Nombre de folds CV")
    args = parser.parse_args()

    print("\n=== Football AI Predictor — Entraînement v3.0 ===\n")

    path = "data/processed/features.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} introuvable → lance feature_engineering.py")

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.dropna(subset=["target"])
    print(f"Dataset : {len(df)} matchs · {df['date'].min().date()} → {df['date'].max().date()}\n")

    params = XGB_PARAMS
    if args.tune:
        print("=== GridSearch Temporel ===\n")
        params = temporal_grid_search(df)

    print(f"=== Validation Croisée ({args.splits} folds) ===\n")
    model, metrics, medians = train_with_cv(df, n_splits=args.splits, params=params)

    if args.shap:
        X, _, _ = prepare_X_y(df)
        run_shap_analysis(model, X)

    if args.lstm:
        export_lstm_sequences(df)

    save_model(model, medians, metrics)

    print("\n✅ Entraînement terminé.")
    print('   python src/predictor.py --home "PSG" --away "Marseille"\n')