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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────────────────────────────────────
# Features 
# ──────────────────────────────────────────────────────────────────────────────
FEATURE_COLS: list[str] = [
    "home_goals_avg",
    "away_goals_avg",
    "home_goals_conceded_avg",
    "away_goals_conceded_avg",
    "home_form",
    "away_form",
    "goals_diff",
    "defense_diff",
    "form_diff",
    "classement_diff",
    "home_shots_target",
    "away_shots_target",
    "h2h_home_win_rate",
    "strength_symmetry",
    "draw_prior",
]

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparamètres par défaut (post-optimisation manuelle)
# ──────────────────────────────────────────────────────────────────────────────
XGB_PARAMS: dict = {
    "n_estimators":      500,
    "learning_rate":     0.03,
    "max_depth":         4,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  5,
    "gamma":             0.1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "objective":         "multi:softprob",
    "num_class":         3,
    "random_state":      42,
    "eval_metric":       "mlogloss",
    "n_jobs":            -1,
}

# Poids de classe pour compenser le déséquilibre du nul (~25% des matchs)
# Dom=0, Nul=1, Ext=2  →  calculés dynamiquement dans train_with_cv()
USE_CLASS_WEIGHTS = True

# Grille de recherche (uniquement activée avec --tune)
PARAM_GRID: dict = {
    "learning_rate":   [0.01, 0.03, 0.05],
    "max_depth":       [3, 4, 5],
    "subsample":       [0.7, 0.8, 0.9],
    "min_child_weight":[3, 5, 7],
}


# ──────────────────────────────────────────────────────────────────────────────
# Préparation des données
# ──────────────────────────────────────────────────────────────────────────────

def add_draw_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute les features spécifiques au signal nul (v2.2).

    RÈGLE ANTI-LEAKAGE STRICTE : toutes les normalisations utilisent des
    constantes domaine fixes (jamais de .max() / .mean() sur le dataset).
    Ces constantes sont des bornes théoriques connues a priori :
      - goals_diff ∈ [-3, 3] en moyenne sur rolling window
      - form_sum   ∈ [0, 30] (10 matchs × 3 pts max)
      - classement_diff ∈ [-1, 1] (déjà normalisé dans feature_engineering)
    """
    df = features_df.copy()

    # 1. Symétrie de force : constante domaine fixe, pas de .max() sur le dataset
    #    goals_diff rolling typiquement dans [-2.5, 2.5] → on sature à 3.0
    GOAL_DIFF_SCALE = 3.0
    df["strength_symmetry"] = (
        1.0 - (df["goals_diff"].abs() / GOAL_DIFF_SCALE).clip(0, 1)
    )

    # 2. Draw prior : forme basse des deux équipes (constante fixe 30 pts max)
    MAX_FORM = 30.0
    df["draw_prior"] = 1.0 - (
        (df["home_form"].clip(0, MAX_FORM) + df["away_form"].clip(0, MAX_FORM))
        / (2 * MAX_FORM)
    )

    # 3. Low-stakes proxy : classement_diff déjà dans [-1,1], form_diff borné
    FORM_DIFF_SCALE = 30.0
    df["low_stakes_proxy"] = (
        (1.0 - df["classement_diff"].abs().clip(0, 1)) *
        (1.0 - (df["form_diff"].abs() / FORM_DIFF_SCALE).clip(0, 1))
    )

    return df


def prepare_X_y(features_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrait X et y depuis le DataFrame de features.
    Les NaN sont remplacés par la médiane de chaque colonne (cohérent
    entre train et inference si on sauvegarde les médianes).
    """
    features_df = add_draw_features(features_df)

    available_cols = [c for c in FEATURE_COLS if c in features_df.columns]
    missing = set(FEATURE_COLS) - set(available_cols)
    if missing:
        print(f"  ⚠️  Features manquantes (seront imputées à 0) : {missing}")

    X_df = features_df.reindex(columns=FEATURE_COLS)
    medians = X_df.median()
    X = X_df.fillna(medians).values
    y = features_df["target"].astype(int).values

    return X, y, medians


# ──────────────────────────────────────────────────────────────────────────────
# Validation croisée temporelle
# ──────────────────────────────────────────────────────────────────────────────

def train_with_cv(
    features_df: pd.DataFrame,
    n_splits: int = 5,
    params: dict | None = None,
) -> tuple[XGBClassifier, dict]:
    """
    Entraîne via TimeSeriesSplit et retourne le modèle final + métriques.
    """
    if params is None:
        params = XGB_PARAMS

    X, y, medians = prepare_X_y(features_df)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    scores_acc, scores_loss = [], []
    baseline = (y == 2).mean()

    print(f"  Baseline naïve (toujours 'dom gagne') : {baseline:.3f}")
    print(f"  {'─'*55}")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        model = XGBClassifier(**params)
        # Poids de classe : surpondère les nuls (classe 1) pour corriger le déséquilibre
        if USE_CLASS_WEIGHTS:
            classes = np.array([0, 1, 2])
            cw = {0: 1.0, 1: 1.4, 2: 1.0}
            sw = np.array([cw[yi] for yi in y[train_idx]])
        else:
            sw = None

        model.fit(
            X[train_idx], y[train_idx],
            sample_weight=sw,
            eval_set=[(X[test_idx], y[test_idx])],
            verbose=False,
        )
        y_pred  = model.predict(X[test_idx])
        y_proba = model.predict_proba(X[test_idx])

        acc  = accuracy_score(y[test_idx], y_pred)
        loss = log_loss(y[test_idx], y_proba)
        scores_acc.append(acc)
        scores_loss.append(loss)

        delta = acc - baseline
        sign  = "▲" if delta > 0 else "▼"
        print(
            f"  Fold {fold} | Acc {acc:.3f} ({sign}{abs(delta):.3f} vs baseline)"
            f" | LogLoss {loss:.3f}"
            f" | {len(train_idx):4d} train / {len(test_idx):4d} test"
        )

    mean_acc  = np.mean(scores_acc)
    mean_loss = np.mean(scores_loss)
    print(f"  {'─'*55}")
    print(f"  Accuracy moyenne : {mean_acc:.3f} ± {np.std(scores_acc):.3f}")
    print(f"  Log Loss moyenne : {mean_loss:.3f} ± {np.std(scores_loss):.3f}")

    # Modèle final sur toutes les données
    print("\n  Entraînement du modèle final (100% des données)…")
    final_model = XGBClassifier(**params)
    if USE_CLASS_WEIGHTS:
        classes = np.array([0, 1, 2])
        cw = compute_class_weight("balanced", classes=classes, y=y)
        sw_final = np.array([cw[yi] for yi in y])
    else:
        sw_final = None
    final_model.fit(X, y, sample_weight=sw_final, verbose=False)

    metrics = {
        "accuracy_mean":  round(mean_acc,  4),
        "accuracy_std":   round(float(np.std(scores_acc)), 4),
        "logloss_mean":   round(mean_loss, 4),
        "logloss_std":    round(float(np.std(scores_loss)), 4),
        "baseline":       round(float(baseline), 4),
        "n_matches":      int(len(y)),
        "features":       FEATURE_COLS,
        "params":         params,
    }

    return final_model, metrics, medians


# ──────────────────────────────────────────────────────────────────────────────
# GridSearch temporellement cohérent
# ──────────────────────────────────────────────────────────────────────────────

def temporal_grid_search(
    features_df: pd.DataFrame,
    param_grid: dict = PARAM_GRID,
    n_splits: int = 4,
) -> dict:
    """
    GridSearch où chaque combinaison est évaluée via TimeSeriesSplit.
    Retourne les meilleurs hyperparamètres selon la Log Loss moyenne.
    """
    from itertools import product

    X, y, _ = prepare_X_y(features_df)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(product(*values))

    print(f"  GridSearch : {len(combos)} combinaisons × {n_splits} folds…\n")

    best_loss   = float("inf")
    best_params = {}

    for i, combo in enumerate(combos):
        params = {**XGB_PARAMS, **dict(zip(keys, combo))}
        losses = []

        for train_idx, test_idx in tscv.split(X):
            m = XGBClassifier(**params)
            m.fit(X[train_idx], y[train_idx], verbose=False)
            losses.append(log_loss(y[test_idx], m.predict_proba(X[test_idx])))

        mean_loss = np.mean(losses)
        print(f"  [{i+1:3d}/{len(combos)}] {dict(zip(keys, combo))} → LogLoss {mean_loss:.4f}")

        if mean_loss < best_loss:
            best_loss   = mean_loss
            best_params = params

    print(f"\n  ✅ Meilleurs params (LogLoss {best_loss:.4f}) : {best_params}")
    return best_params


# ──────────────────────────────────────────────────────────────────────────────
# Analyse SHAP
# ──────────────────────────────────────────────────────────────────────────────

def run_shap_analysis(model: XGBClassifier, X: np.ndarray, output_dir: str = "models") -> None:
    """
    Calcule et affiche les valeurs SHAP moyennes (feature importance globale).
    Identifie les features candidates à la suppression (SHAP < seuil).
    """
    try:
        import shap
    except ImportError:
        print("  ⚠️  shap non installé. pip install shap")
        return

    print("\n  Calcul des valeurs SHAP (échantillon 500 matchs)…")
    sample_size = min(500, len(X))
    X_sample    = X[:sample_size]

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Gestion des formats selon la version de SHAP :
    #   - SHAP < 0.40  : liste de matrices (n_samples, n_features) x n_classes
    #   - SHAP >= 0.40 : array 3D (n_samples, n_features, n_classes)
    if isinstance(shap_values, list):
        # Ancien format : liste de 3 matrices 2D
        mean_abs_shap = np.mean(
            [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
        )
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        # Nouveau format : (n_samples, n_features, n_classes)
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
    else:
        # Cas binaire ou fallback
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Garantir un vecteur 1D avant de construire le DataFrame
    mean_abs_shap = np.array(mean_abs_shap).flatten()

    importance_df = pd.DataFrame({
        "feature":    FEATURE_COLS[:len(mean_abs_shap)],
        "shap_mean":  mean_abs_shap,
    }).sort_values("shap_mean", ascending=False)

    print(f"\n  {'─'*50}")
    print(f"  {'Feature':<35} {'SHAP moyen':>10}")
    print(f"  {'─'*50}")
    for _, row in importance_df.iterrows():
        bar   = "█" * int(row["shap_mean"] / importance_df["shap_mean"].max() * 25)
        flag  = " ← bruit?" if row["shap_mean"] < 0.005 else ""
        print(f"  {row['feature']:<35} {row['shap_mean']:>8.4f}  {bar}{flag}")

    # Sauvegarde CSV
    shap_path = os.path.join(output_dir, "shap_importance.csv")
    importance_df.to_csv(shap_path, index=False)
    print(f"\n  ✅ SHAP sauvegardé : {shap_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Sauvegarde
# ──────────────────────────────────────────────────────────────────────────────

def save_model(
    model: XGBClassifier,
    medians: pd.Series,
    metrics: dict,
    output_dir: str = "models",
) -> None:
    """Sauvegarde modèle, médianes d'imputation, features et métriques."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    joblib.dump(model,   os.path.join(output_dir, "xgb_model.pkl"))
    joblib.dump(medians, os.path.join(output_dir, "feature_medians.pkl"))

    with open(os.path.join(output_dir, "feature_names.txt"), "w") as f:
        f.write("\n".join(FEATURE_COLS))

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  ✅ Modèle       : {output_dir}/xgb_model.pkl")
    print(f"  ✅ Médianes     : {output_dir}/feature_medians.pkl")
    print(f"  ✅ Métriques    : {output_dir}/metrics.json")


# ──────────────────────────────────────────────────────────────────────────────
# Entrée principale
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune",  action="store_true", help="Activer GridSearch")
    parser.add_argument("--shap",  action="store_true", help="Activer analyse SHAP")
    parser.add_argument("--splits", type=int, default=5, help="Nombre de folds CV")
    args = parser.parse_args()

    print("\n=== Entraînement Football AI Predictor v2.1 ===\n")

    input_path = "data/processed/features.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"{input_path} introuvable. Lance d'abord : python src/feature_engineering.py"
        )

    features_df = pd.read_csv(input_path, parse_dates=["date"])
    features_df = features_df.dropna(subset=["target"])
    print(f"Features chargées : {len(features_df)} matchs\n")

    params = XGB_PARAMS
    if args.tune:
        print("=== GridSearch temporel ===\n")
        params = temporal_grid_search(features_df)

    print(f"=== Validation croisée ({args.splits} folds) ===\n")
    model, metrics, medians = train_with_cv(features_df, n_splits=args.splits, params=params)

    if args.shap:
        X, _, _ = prepare_X_y(features_df)
        run_shap_analysis(model, X)

    save_model(model, medians, metrics)

    print("\n✅ Pipeline terminé. Prédiction :")
    print('   python src/predictor.py --home "PSG" --away "Marseille"\n')