"""Tests unitaires de tabicl_gui/utils.py.

Aucune dépendance tabicl ni LM Studio requise.
"""
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tabicl_gui.utils import (
    compute_classification_metrics,
    compute_regression_metrics,
    detect_task_type,
    load_data,
)


# ═══════════════════════════════════════════════════════════════════════
# load_data
# ═══════════════════════════════════════════════════════════════════════

class TestLoadData:
    def test_csv_local(self, tmp_csv, sample_df_clf):
        df = load_data(tmp_csv)
        assert df.shape == sample_df_clf.shape
        assert list(df.columns) == list(sample_df_clf.columns)

    def test_excel_local(self, tmp_excel, sample_df_clf):
        df = load_data(tmp_excel)
        assert df.shape == sample_df_clf.shape
        assert list(df.columns) == list(sample_df_clf.columns)

    def test_json_local(self, tmp_json, sample_df_clf):
        df = load_data(tmp_json)
        assert df.shape == sample_df_clf.shape

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            load_data("/nonexistent/path/file.csv")

    def test_url_csv(self, sample_df_clf):
        csv_bytes = sample_df_clf.to_csv(index=False).encode()
        mock_resp = MagicMock()
        mock_resp.content = csv_bytes
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.get", return_value=mock_resp):
            df = load_data("https://example.com/data.csv")
        assert df.shape == sample_df_clf.shape

    def test_url_excel(self, sample_df_clf):
        buf = io.BytesIO()
        sample_df_clf.to_excel(buf, index=False)
        mock_resp = MagicMock()
        mock_resp.content = buf.getvalue()
        mock_resp.raise_for_status = MagicMock()
        with patch("requests.get", return_value=mock_resp):
            df = load_data("https://example.com/data.xlsx")
        assert df.shape == sample_df_clf.shape

    def test_url_connection_error_raises(self):
        with patch("requests.get", side_effect=ConnectionError("no network")):
            with pytest.raises(Exception):
                load_data("https://unreachable.invalid/data.csv")

    def test_url_http_error_raises(self):
        import requests
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404")
        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(Exception):
                load_data("https://example.com/missing.csv")

    def test_leading_whitespace_stripped(self, tmp_csv):
        df = load_data(f"  {tmp_csv}  ")
        assert len(df) > 0


# ═══════════════════════════════════════════════════════════════════════
# detect_task_type
# ═══════════════════════════════════════════════════════════════════════

class TestDetectTaskType:
    def test_bool_series(self):
        s = pd.Series([True, False, True, False] * 10)
        assert detect_task_type(s) == "classification"

    def test_int_few_uniques(self):
        # 5 valeurs uniques sur 50 lignes → classification
        s = pd.Series(np.tile([0, 1, 2, 3, 4], 10))
        assert detect_task_type(s) == "classification"

    def test_float_many_uniques(self):
        # 50 valeurs uniques sur 50 lignes → regression
        s = pd.Series(np.linspace(0.0, 100.0, 50))
        assert detect_task_type(s) == "regression"

    def test_string_series(self):
        s = pd.Series(["cat", "dog", "bird"] * 17)
        assert detect_task_type(s) == "classification"

    def test_exactly_20_uniques(self):
        # 20 uniques ≤ 20 → classification
        s = pd.Series(list(range(20)) * 3)
        assert detect_task_type(s) == "classification"

    def test_21_uniques_exact_5pct_proportion(self):
        # 21/420 = exactement 5% — la condition est < 0.05 (strict)
        # → n_unique (21) > 20 ET 0.05 n'est pas < 0.05 → regression
        s = pd.Series(list(range(21)) * 20)
        assert detect_task_type(s) == "regression"

    def test_20_uniques_below_5pct(self):
        # 20/420 < 5% ET ≤ 20 → classification (double condition vraie)
        s = pd.Series(list(range(20)) * 21)
        assert detect_task_type(s) == "classification"

    def test_many_uniques_large_proportion(self):
        # 100 uniques sur 100 lignes → regression
        s = pd.Series(range(100), dtype=float)
        assert detect_task_type(s) == "regression"


# ═══════════════════════════════════════════════════════════════════════
# compute_classification_metrics
# ═══════════════════════════════════════════════════════════════════════

class TestClassificationMetrics:
    def _perfect_binary(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        return y, y.copy()

    def _wrong_binary(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = 1 - y_true          # tout faux
        return y_true, y_pred

    def test_perfect_accuracy(self):
        y, p = self._perfect_binary()
        m = compute_classification_metrics(y, p)
        assert m["accuracy"] == pytest.approx(1.0)

    def test_perfect_f1(self):
        y, p = self._perfect_binary()
        m = compute_classification_metrics(y, p)
        assert m["f1"] == pytest.approx(1.0)

    def test_zero_accuracy(self):
        y, p = self._wrong_binary()
        m = compute_classification_metrics(y, p)
        assert m["accuracy"] == pytest.approx(0.0)

    def test_roc_auc_present_with_proba(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0, 0, 1, 1])
        proba = np.column_stack([1 - p, p]).astype(float)
        m = compute_classification_metrics(y, p, proba)
        assert "roc_auc" in m
        assert m["roc_auc"] == pytest.approx(1.0)

    def test_roc_auc_absent_without_proba(self):
        y, p = self._perfect_binary()
        m = compute_classification_metrics(y, p)
        assert "roc_auc" not in m

    def test_multiclass_three_classes(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        p = np.array([0, 1, 2, 0, 1, 2])
        m = compute_classification_metrics(y, p)
        assert len(m["classes"]) == 3

    def test_confusion_matrix_shape(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        p = np.array([0, 1, 2, 0, 1, 2])
        m = compute_classification_metrics(y, p)
        assert m["confusion_matrix"].shape == (3, 3)

    def test_report_is_string(self):
        y, p = self._perfect_binary()
        m = compute_classification_metrics(y, p)
        assert isinstance(m["report"], str)
        assert len(m["report"]) > 0


# ═══════════════════════════════════════════════════════════════════════
# compute_regression_metrics
# ═══════════════════════════════════════════════════════════════════════

class TestRegressionMetrics:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = compute_regression_metrics(y, y.copy())
        assert m["rmse"] == pytest.approx(0.0)
        assert m["mae"] == pytest.approx(0.0)
        assert m["r2"] == pytest.approx(1.0)

    def test_zero_predictions_negative_r2(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p = np.zeros_like(y)
        m = compute_regression_metrics(y, p)
        assert m["r2"] < 0

    def test_rmse_formula(self):
        y    = np.array([0.0, 0.0, 0.0])
        pred = np.array([1.0, 2.0, 3.0])
        m = compute_regression_metrics(y, pred)
        expected_rmse = float(np.sqrt(np.mean([1, 4, 9])))
        assert m["rmse"] == pytest.approx(expected_rmse)

    def test_all_keys_present(self):
        y = np.array([1.0, 2.0, 3.0])
        m = compute_regression_metrics(y, y)
        for key in ("rmse", "mae", "r2", "mse"):
            assert key in m

    def test_rmse_equals_sqrt_mse(self):
        rng = np.random.default_rng(7)
        y = rng.standard_normal(30)
        p = rng.standard_normal(30)
        m = compute_regression_metrics(y, p)
        assert m["rmse"] == pytest.approx(np.sqrt(m["mse"]))
