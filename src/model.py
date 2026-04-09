"""
model.py
--------
Entraînement, validation et sauvegarde du modèle XGBoost.

Stratégie de validation : TimeSeriesSplit (validation croisée temporelle)
— on entraîne toujours sur le passé, on valide sur le futur immédiat.
Jamais de split aléatoire sur des données temporelles !

Utilisation :
    python src/model.py

Entrée :
    data/processed/features.csv

Sortie :
    models/xgb_model.pkl     — modèle entraîné
    models/feature_names.txt — liste des features utilisées
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, log_loss
import joblib
import os


# Colonnes utilisées comme features (dans l'ordre)
FEATURE_COLS = [
    "home_goals_avg",
    "away_goals_avg",
    "home_goals_conceded_avg",
    "away_goals_conceded_avg",
    "home_form",
    "away_form",
    "goals_diff",
    "defense_diff",
    "form_diff",
    "home_shots_target",
    "away_shots_target",
]

# Hyperparamètres du modèle
XGB_PARAMS = {
    "n_estimators":    300,
    "learning_rate":   0.05,
    "max_depth":       4,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "objective":       "multi:softprob",
    "num_class":       3,
    "random_state":    42,
    "eval_metric":     "mlogloss",
}


def train_with_cv(features_df, n_splits=5):
    """
    Entraîne le modèle avec validation croisée temporelle.

    Args:
        features_df (pd.DataFrame): Features construites par feature_engineering.py
        n_splits (int): Nombre de folds de validation croisée

    Returns:
        tuple: (modèle final entraîné, liste des scores par fold)
    """
    # Préparer X et y
    X = features_df[FEATURE_COLS].fillna(features_df[FEATURE_COLS].median()).values
    y = features_df["target"].astype(int).values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores_acc  = []
    scores_loss = []

    print(f"=== Validation croisée temporelle ({n_splits} folds) ===\n")

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        y_pred       = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        acc  = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_pred_proba)

        scores_acc.append(acc)
        scores_loss.append(loss)

        print(f"  Fold {fold} — Accuracy: {acc:.3f} | Log Loss: {loss:.3f}"
              f"  ({len(train_idx)} train / {len(test_idx)} test)")

    print(f"\n  Accuracy moyenne : {np.mean(scores_acc):.3f} ± {np.std(scores_acc):.3f}")
    print(f"  Log Loss moyenne : {np.mean(scores_loss):.3f} ± {np.std(scores_loss):.3f}")
    print(f"  Baseline naïve   : {(y == 2).mean():.3f} (toujours prédire 'dom gagne')")

    # Modèle final entraîné sur toutes les données
    print("\n=== Entraînement du modèle final (toutes données) ===")
    final_model = XGBClassifier(**XGB_PARAMS)
    final_model.fit(X, y)

    return final_model, scores_acc


def save_model(model, output_dir="models"):
    """Sauvegarde le modèle et la liste des features."""
    os.makedirs(output_dir, exist_ok=True)

    model_path   = os.path.join(output_dir, "xgb_model.pkl")
    feature_path = os.path.join(output_dir, "feature_names.txt")

    joblib.dump(model, model_path)

    with open(feature_path, "w") as f:
        f.write("\n".join(FEATURE_COLS))

    print(f"\n✅ Modèle sauvegardé : {model_path}")
    print(f"✅ Features sauvegardées : {feature_path}")


def print_feature_importance(model):
    """Affiche les features les plus importantes selon XGBoost."""
    importance = model.get_booster().get_fscore()
    if not importance:
        return

    importance_df = pd.DataFrame(
        list(importance.items()), columns=["feature", "importance"]
    ).sort_values("importance", ascending=False)

    print("\n=== Importance des features ===\n")
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["importance"] / importance_df["importance"].max() * 30)
        print(f"  {row['feature']:<30} {bar}")


if __name__ == "__main__":
    print("\n=== Entraînement du modèle Football AI Predictor ===\n")

    input_path = "data/processed/features.csv"
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"{input_path} introuvable. Lance d'abord : python src/feature_engineering.py"
        )

    features_df = pd.read_csv(input_path, parse_dates=["date"])
    features_df = features_df.dropna(subset=["target"])
    print(f"Features chargées : {len(features_df)} matchs\n")

    model, scores = train_with_cv(features_df)

    print_feature_importance(model)
    save_model(model)

    print("\n✅ Pipeline terminé. Lance la prédiction avec :")
    print('   python src/predictor.py --home "PSG" --away "Marseille"\n')
