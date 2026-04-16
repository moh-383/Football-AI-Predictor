# ⚽ Football AI Predictor

> Système de Machine Learning pour la prédiction des issues de matchs de football  
> **Stack :** Python · XGBoost · Scikit-learn · Pandas · LSTM (PyTorch)

---

## 📋 Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Prérequis](#-prérequis)
- [Installation](#-installation)
- [Structure du projet](#-structure-du-projet)
- [Utilisation rapide](#-utilisation-rapide)
- [Pipeline complet](#-pipeline-complet)
- [Workflow Git (collaboration)](#-workflow-git-collaboration)
- [Résultats attendus](#-résultats-attendus)
- [Roadmap](#-roadmap)
- [Contribuer](#-contribuer)

---

## 🎯 Vue d'ensemble

Ce projet prédit les probabilités des trois issues d'un match de football :

| Issue | Encodage | Exemple |
|-------|----------|---------|
| Victoire domicile | `2` | PSG gagne à Paris |
| Match nul | `1` | 1-1 |
| Victoire extérieur | `0` | Lyon gagne à Paris |

**Ce que le système produit concrètement :**

```
>>> python src/predictor.py --home "PSG" --away "Lyon"

=== Prédiction : PSG vs Lyon ===
Victoire PSG   :  62.3%  ████████████
Match nul      :  21.1%  ████
Victoire Lyon  :  16.6%  ███

Score le plus probable : 2 - 0
Expected Goals  →  PSG: 1.82  |  Lyon: 0.71
```

**Données utilisées :** Ligue 1 française (5 saisons), source [football-data.co.uk](https://www.football-data.co.uk/)  
**Précision typique :** 58–65% sur les issues 1/N/2 (baseline naïve ≈ 45%)

---

## 🖥️ Prérequis

| Outil | Version minimale | Vérification |
|-------|-----------------|--------------|
| Python | 3.9+ | `python --version` |
| pip | 21.0+ | `pip --version` |
| Git | 2.30+ | `git --version` |

> **IDE recommandé :** [VS Code](https://code.visualstudio.com/) avec les extensions `Python`, `Pylance`, `Jupyter`, `GitLens`

---

## ⚙️ Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/moh-383/football-predictor.git
cd football-predictor
```

### 2. Créer et activer l'environnement virtuel

```bash
# Créer l'environnement (une seule fois)
python -m venv football_env

# Activer — Windows
football_env\Scripts\activate

# Activer — macOS / Linux
source football_env/bin/activate

# ✅ Vérification : ton prompt doit afficher (football_env)
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Vérifier l'installation

```bash
python scripts/check_install.py
```

Résultat attendu :
```
✅ pandas       2.x.x
✅ numpy        1.x.x
✅ scipy        1.x.x
✅ scikit-learn 1.x.x
✅ xgboost      2.x.x
✅ shap         0.x.x
✅ matplotlib   3.x.x
✅ joblib       1.x.x
✅ requests     2.x.x
Tout est installé correctement !
```

---

## 📁 Structure du projet

```
football-predictor/
│
├── 📂 data/
│   ├── raw/                  # CSV bruts téléchargés (ne jamais modifier)
│   │   └── ligue1_2324.csv
│   └── processed/            # Données nettoyées et features construites
│       ├── ligue1_clean.csv
│       └── features.csv
│
├── 📂 src/                   # Code source principal
│   ├── data_loader.py        # Téléchargement + nettoyage des données brutes
│   ├── feature_engineering.py # Construction de toutes les variables du modèle
│   ├── model.py              # Entraînement, validation croisée, sauvegarde
│   └── predictor.py          # Interface de prédiction (point d'entrée principal)
│
├── 📂 notebooks/             # Analyses exploratoires Jupyter
│   ├── 01_exploration.ipynb  # Visualisation des données brutes
│   ├── 02_features.ipynb     # Analyse et validation des features
│   └── 03_model_analysis.ipynb # Performances, erreurs, SHAP
│
├── 📂 models/                # Modèles entraînés sauvegardés
│   └── xgb_model.pkl         # Généré automatiquement par model.py
│
├── 📂 tests/                 # Tests unitaires
│   ├── test_data_loader.py
│   ├── test_features.py
│   └── test_model.py
│
├── 📂 scripts/               # Scripts utilitaires
│   └── check_install.py      # Vérifie que l'environnement est correct
│
├── .gitignore                # Fichiers exclus du suivi Git
├── requirements.txt          # Dépendances Python
└── README.md                 # Ce fichier
```

> **Règle d'or :** Ne jamais modifier les fichiers dans `data/raw/`. Toute transformation se fait dans les scripts `src/`.

---

## 🚀 Utilisation rapide

### Étape 1 : Télécharger les données

```bash
python src/data_loader.py
```

Ce script télécharge automatiquement les 5 dernières saisons de Ligue 1 et les sauvegarde dans `data/raw/`.

### Étape 2 : Construire les features

```bash
python src/feature_engineering.py
```

Génère `data/processed/features.csv` avec toutes les variables du modèle.

### Étape 3 : Entraîner le modèle

```bash
python src/model.py
```

Lance la validation croisée temporelle, affiche les métriques, et sauvegarde le modèle dans `models/xgb_model.pkl`.

### Étape 4 : Prédire un match

```bash
python src/predictor.py --home "Paris SG" --away "Marseille"
```

### Tout en une commande (pipeline complet)

```bash
python scripts/run_pipeline.py
```

---

## 🔄 Pipeline complet

```
football-data.co.uk
       │
       ▼
data_loader.py          → data/raw/*.csv           (données brutes)
       │
       ▼
feature_engineering.py  → data/processed/features.csv  (features construites)
       │
       ▼
model.py                → models/xgb_model.pkl      (modèle entraîné)
       │
       ▼
predictor.py            → probabilités + score attendu
```

---

## 🤝 Workflow Git (collaboration)

Ce projet est développé à deux. Voici les règles de collaboration pour éviter les conflits.

### Branches

| Branche | Rôle |
|---------|------|
| `main` | Code stable, testé : ne jamais pousser directement |
| `dev` | Branche d'intégration principale |
| `feature/nom` | Nouvelle fonctionnalité en cours |
| `fix/nom` | Correction de bug |

### Démarrer une nouvelle tâche

```bash
# 1. Se mettre à jour depuis dev
git checkout dev
git pull origin dev

# 2. Créer ta branche de travail
git checkout -b feature/ma-fonctionnalite

# 3. Travailler, committer régulièrement
git add src/mon_fichier.py
git commit -m "feat: ajouter rolling features pour la forme à domicile"

# 4. Pousser et ouvrir une Pull Request vers dev
git push origin feature/ma-fonctionnalite
```

### Convention des messages de commit

```
feat:     Nouvelle fonctionnalité
fix:      Correction de bug
data:     Modification liée aux données
model:    Changement du modèle ou des hyperparamètres
docs:     Documentation uniquement
refactor: Refactoring sans changement de comportement
test:     Ajout ou modification de tests
```

**Exemples :**
```bash
git commit -m "feat: ajouter feature h2h_win_rate"
git commit -m "fix: corriger data leakage dans rolling window"
git commit -m "model: optimiser hyperparamètres XGBoost learning_rate=0.03"
git commit -m "data: ajouter saisons 2019-2020 Ligue 1"
```

### Règles importantes

- ✅ Toujours travailler sur une branche `feature/` ou `fix/`
- ✅ Pull Request obligatoire pour merger dans `dev`
- ✅ L'autre personne relit le code avant de merger (code review)
- ❌ Ne jamais pousser directement sur `main`
- ❌ Ne jamais committer les fichiers `data/raw/` (trop lourds, dans .gitignore)
- ❌ Ne jamais committer `models/*.pkl` ni `football_env/`

---

## 📊 Résultats attendus

| Métrique | Baseline naïve | Notre modèle (cible) |
|----------|---------------|----------------------|
| Accuracy | ~45% | 58 – 65% |
| Log Loss | ~1.05 | 0.92 – 1.00 |
| ROC-AUC | ~0.50 | 0.63 – 0.70 |

> La baseline naïve correspond à toujours prédire "victoire domicile" (issue la plus fréquente ≈ 45%).

---

## 🗓️ Roadmap

- [x] Structure du projet et README
- [x] **Phase 1** — Collecte et exploration des données (S1–S2)
- [x] **Phase 2** — Feature engineering complet (S3–S4)
- [ ] **Phase 3** — Entraînement XGBoost + validation (S5–S6)
- [ ] **Phase 4** — Interface de prédiction (S7–S8)
- [ ] **Phase 5** — Extension LSTM sur séquences (S9–S10)
- [ ] **Phase 6** — Pipeline automatisé + API temps réel (S11–S12)

---

## 👥 Contribuer

Ce projet est développé en collaboration privée. Pour toute question :

1. Ouvrir une **Issue** GitHub avec le tag approprié (`bug`, `enhancement`, `question`)
2. Discuter avant de coder pour éviter les doublons de travail
3. Toute Pull Request doit passer les tests : `python -m pytest tests/`

---

*Document de référence complet disponible dans le repo : `docs/Football_AI_Predictor_Reference.docx`*
