"""
run_pipeline.py — Football AI Predictor v3.0
=============================================
Lance le pipeline complet :
    1. Téléchargement données (10 saisons)
    2. Feature engineering v3.0
    3. Entraînement XGBoost + CV temporelle
    4. Backtesting 200 matchs
    5. (optionnel) SHAP + export LSTM

Usage :
    python scripts/run_pipeline.py [--shap] [--lstm] [--tune]
"""

import argparse
import subprocess
import sys
import time


def run(cmd: str, desc: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {desc}")
    print(f"{'═'*60}\n")
    t0     = time.time()
    result = subprocess.run(cmd, shell=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n❌ Erreur dans : {cmd}")
        sys.exit(result.returncode)
    print(f"\n  ⏱  {elapsed:.1f}s\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline complet Football AI v3.0")
    parser.add_argument("--shap",  action="store_true", help="Analyse SHAP")
    parser.add_argument("--lstm",  action="store_true", help="Export séquences LSTM")
    parser.add_argument("--tune",  action="store_true", help="GridSearch hyperparamètres")
    parser.add_argument("--n_bt",  type=int, default=200, help="Matchs de backtest")
    args = parser.parse_args()

    print("\n🚀 Football AI Predictor v3.0 — Pipeline Complet\n")

    run("python src/data_loader.py",         "1/5 · Téléchargement données (10 saisons)")
    run("python src/feature_engineering.py", "2/5 · Feature Engineering v3.0")

    model_cmd = "python src/model.py"
    if args.tune:
        model_cmd += " --tune"
    if args.shap:
        model_cmd += " --shap"
    if args.lstm:
        model_cmd += " --lstm"
    run(model_cmd, "3/5 · Entraînement XGBoost (CV temporelle)")

    bt_cmd = f"python src/backtester.py --n {args.n_bt} --report"
    if args.shap:
        bt_cmd += " --shap"
    run(bt_cmd, f"4/5 · Backtesting ({args.n_bt} matchs)")

    print(f"\n{'═'*60}")
    print("  ✅ Pipeline v3.0 terminé !")
    print(f"{'═'*60}")
    print("\n  Prédiction d'un match :")
    print('  python src/predictor.py --home "Paris SG" --away "Marseille"\n')