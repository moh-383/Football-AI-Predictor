"""
Tests unitaires pour data_loader.py
Lancer avec : python -m pytest tests/
"""
import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_loader import clean_data


def make_dummy_df():
    """Crée un DataFrame de test minimal."""
    return pd.DataFrame({
        "Date": ["01/09/2023", "15/09/2023", "30/09/2023"],
        "HomeTeam": ["PSG", "Lyon", "Monaco"],
        "AwayTeam": ["Marseille", "Nice", "Lens"],
        "FTHG": [2, 1, 0],
        "FTAG": [0, 1, 2],
        "FTR":  ["H", "D", "A"],
        "HS": [12, 8, 5], "AS": [5, 6, 10],
        "HST": [6, 4, 2], "AST": [2, 3, 5],
        "HC": [5, 3, 2],  "AC": [2, 4, 6],
        "HF": [8, 10, 7], "AF": [9, 8, 11],
        "HY": [1, 2, 0],  "AY": [0, 1, 2],
        "HR": [0, 0, 0],  "AR": [0, 0, 0],
        "season": ["2324", "2324", "2324"],
    })


def test_clean_data_columns():
    df = clean_data(make_dummy_df())
    assert "date" in df.columns
    assert "target" in df.columns
    assert "home_team" in df.columns


def test_clean_data_target_encoding():
    df = clean_data(make_dummy_df())
    # H → 2, D → 1, A → 0
    assert set(df["target"].dropna().unique()).issubset({0, 1, 2})


def test_clean_data_sorted_by_date():
    df = clean_data(make_dummy_df())
    assert df["date"].is_monotonic_increasing


def test_clean_data_no_null_targets():
    df = clean_data(make_dummy_df())
    assert df["target"].notna().all()