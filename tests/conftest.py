"""
Fixtures partagées pour toute la suite de tests TabICL GUI.

• QApplication singleton (scope session) — PyQt6 ne tolère qu'une seule instance.
• DataFrames reproductibles (clf, reg, avec NaN).
• Fichiers temporaires CSV/Excel.
• QT_QPA_PLATFORM=offscreen pour fonctionner sans serveur X.
"""
import os
import sys

# Doit être positionné AVANT tout import PyQt6 ou matplotlib
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("Agg")   # backend non-interactif

import io
import numpy as np
import pandas as pd
import pytest
from PyQt6.QtWidgets import QApplication


# ── QApplication ─────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qapp_instance():
    """QApplication unique pour toute la session."""
    existing = QApplication.instance()
    if existing is not None:
        return existing
    return QApplication(sys.argv[:1])


# ── DataFrames ────────────────────────────────────────────────────────

@pytest.fixture
def sample_df_clf():
    """50 lignes, 4 features numériques + cible catégorielle (3 classes)."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "sepal_length": rng.uniform(4.0, 8.0, 50),
        "sepal_width":  rng.uniform(2.0, 4.5, 50),
        "petal_length": rng.uniform(1.0, 7.0, 50),
        "petal_width":  rng.uniform(0.1, 2.5, 50),
        "species":      rng.choice(["setosa", "versicolor", "virginica"], 50),
    })


@pytest.fixture
def sample_df_reg():
    """50 lignes, 3 features + cible continue."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 3))
    return pd.DataFrame({
        "feature_a": X[:, 0],
        "feature_b": X[:, 1],
        "feature_c": X[:, 2],
        "price":     X[:, 0] * 2.0 + X[:, 1] * 0.5 + rng.standard_normal(50) * 0.1,
    })


@pytest.fixture
def sample_df_missing(sample_df_clf):
    """Copie de clf avec ~10 % de valeurs manquantes."""
    df = sample_df_clf.copy()
    rng = np.random.default_rng(42)
    mask = rng.random(df.shape) < 0.1
    df_obj = df.astype(object)
    df_obj[mask] = np.nan
    # Reconstruit avec les dtypes d'origine
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df_obj[col])
        except (ValueError, TypeError):
            df[col] = df_obj[col]
    return df


@pytest.fixture
def state():
    """État partagé vide, comme utilisé par les onglets."""
    return {}


# ── Fichiers temporaires ──────────────────────────────────────────────

@pytest.fixture
def tmp_csv(tmp_path, sample_df_clf):
    p = tmp_path / "data.csv"
    sample_df_clf.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def tmp_excel(tmp_path, sample_df_clf):
    p = tmp_path / "data.xlsx"
    sample_df_clf.to_excel(p, index=False)
    return str(p)


@pytest.fixture
def tmp_json(tmp_path, sample_df_clf):
    p = tmp_path / "data.json"
    sample_df_clf.to_json(p, orient="records")
    return str(p)
