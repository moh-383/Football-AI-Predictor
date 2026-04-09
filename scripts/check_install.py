"""
check_install.py
----------------
Vérifie que toutes les dépendances du projet sont correctement installées.
Utilisation : python scripts/check_install.py
"""

import sys

REQUIRED = [
    ("pandas",      "pandas"),
    ("numpy",       "numpy"),
    ("scipy",       "scipy"),
    ("scikit-learn","sklearn"),
    ("xgboost",     "xgboost"),
    ("lightgbm",    "lightgbm"),
    ("shap",        "shap"),
    ("matplotlib",  "matplotlib"),
    ("seaborn",     "seaborn"),
    ("joblib",      "joblib"),
    ("requests",    "requests"),
]

OPTIONAL = [
    ("torch (PyTorch)", "torch"),
]

def check():
    print("\n=== Vérification de l'environnement Football AI Predictor ===\n")
    all_ok = True

    print("--- Dépendances obligatoires ---")
    for display_name, import_name in REQUIRED:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "?")
            print(f"  ✅ {display_name:<18} {version}")
        except ImportError:
            print(f"  ❌ {display_name:<18} NON INSTALLÉ")
            all_ok = False

    print("\n--- Dépendances optionnelles (Phase 5 — Deep Learning) ---")
    for display_name, import_name in OPTIONAL:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "?")
            print(f"  ✅ {display_name:<22} {version}")
        except ImportError:
            print(f"  ⚠️  {display_name:<22} non installé (optionnel, nécessaire pour le LSTM)")

    print(f"\n--- Python : {sys.version} ---\n")

    if all_ok:
        print("✅ Tout est installé correctement. Tu peux démarrer !\n")
    else:
        print("❌ Des dépendances manquent. Lance : pip install -r requirements.txt\n")
        sys.exit(1)

if __name__ == "__main__":
    check()
