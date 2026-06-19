"""Tests des workers QThread de tabicl_gui/workers.py.

• tabicl est mocké via sys.modules — jamais installé.
• pytest-qt (qtbot) est utilisé pour attendre les signaux.
"""
import sys
import traceback
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tabicl_gui.workers import LLMWorker, TrainWorker


# ── helpers ───────────────────────────────────────────────────────────

def _make_X_y(n=20, f=4):
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, f)), rng.integers(0, 2, n)


def _tabicl_clf_mock(preds=None, probas=None):
    """Retourne un mock TabICLClassifier cohérent."""
    m = MagicMock()
    m.return_value.predict.return_value = preds if preds is not None else np.array([0, 1, 0, 1, 0])
    if probas is not None:
        m.return_value.predict_proba.return_value = probas
    else:
        del m.return_value.predict_proba   # simule l'absence de la méthode
    return m


def _tabicl_reg_mock(preds=None):
    m = MagicMock()
    m.return_value.predict.return_value = preds if preds is not None else np.linspace(0, 1, 10)
    del m.return_value.predict_proba
    return m


# ═══════════════════════════════════════════════════════════════════════
# TrainWorker — classification
# ═══════════════════════════════════════════════════════════════════════

class TestTrainWorkerClassification:
    def test_finished_signal_emitted(self, qtbot, qapp_instance):
        X, y = _make_X_y()
        clf_cls = _tabicl_clf_mock()

        mock_tabicl = MagicMock()
        mock_tabicl.TabICLClassifier = clf_cls

        worker = TrainWorker("classification", {}, X, y, X[:5])
        with patch.dict(sys.modules, {"tabicl": mock_tabicl}):
            with qtbot.waitSignal(worker.finished, timeout=5000) as blocker:
                worker.start()
                worker.wait(5000)

        model, preds, probas = blocker.args
        assert model is not None
        assert len(preds) == 5

    def test_probas_included_when_predict_proba_exists(self, qtbot, qapp_instance):
        X, y = _make_X_y()
        expected_probas = np.column_stack([np.linspace(0, 1, 5), np.linspace(1, 0, 5)])

        clf_cls = MagicMock()
        clf_cls.return_value.predict.return_value = np.array([0] * 5)
        clf_cls.return_value.predict_proba.return_value = expected_probas

        mock_tabicl = MagicMock()
        mock_tabicl.TabICLClassifier = clf_cls

        worker = TrainWorker("classification", {}, X, y, X[:5])
        with patch.dict(sys.modules, {"tabicl": mock_tabicl}):
            with qtbot.waitSignal(worker.finished, timeout=5000) as blocker:
                worker.start()
                worker.wait(5000)

        _, _, probas = blocker.args
        assert probas is not None
        assert probas.shape == (5, 2)

    def test_error_signal_on_fit_failure(self, qtbot, qapp_instance):
        X, y = _make_X_y()

        clf_cls = MagicMock()
        clf_cls.return_value.fit.side_effect = RuntimeError("GPU OOM")

        mock_tabicl = MagicMock()
        mock_tabicl.TabICLClassifier = clf_cls

        worker = TrainWorker("classification", {}, X, y, X[:5])
        with patch.dict(sys.modules, {"tabicl": mock_tabicl}):
            with qtbot.waitSignal(worker.error, timeout=5000) as blocker:
                worker.start()
                worker.wait(5000)

        error_msg = blocker.args[0]
        assert "RuntimeError" in error_msg

    def test_stop_before_start_prevents_finished(self, qapp_instance):
        X, y = _make_X_y()
        clf_cls = _tabicl_clf_mock()
        mock_tabicl = MagicMock()
        mock_tabicl.TabICLClassifier = clf_cls

        worker = TrainWorker("classification", {}, X, y, X[:5])
        received = []
        worker.finished.connect(lambda *args: received.append(True))

        with patch.dict(sys.modules, {"tabicl": mock_tabicl}):
            worker.stop()   # avant start
            worker.start()
            worker.wait(3000)

        assert len(received) == 0


# ═══════════════════════════════════════════════════════════════════════
# TrainWorker — regression
# ═══════════════════════════════════════════════════════════════════════

class TestTrainWorkerRegression:
    def test_finished_signal_regression(self, qtbot, qapp_instance):
        X, y = _make_X_y()
        y = y.astype(float)
        reg_cls = _tabicl_reg_mock(preds=np.linspace(0, 1, 5))

        mock_tabicl = MagicMock()
        mock_tabicl.TabICLRegressor = reg_cls

        worker = TrainWorker("regression", {}, X, y, X[:5])
        with patch.dict(sys.modules, {"tabicl": mock_tabicl}):
            with qtbot.waitSignal(worker.finished, timeout=5000) as blocker:
                worker.start()
                worker.wait(5000)

        _, preds, probas = blocker.args
        assert len(preds) == 5
        assert probas is None


# ═══════════════════════════════════════════════════════════════════════
# LLMWorker
# ═══════════════════════════════════════════════════════════════════════

class TestLLMWorker:
    def _make_client(self, constraints=None, raise_exc=None):
        client = MagicMock()
        if raise_exc:
            client.analyze_columns.side_effect = raise_exc
        else:
            client.analyze_columns.return_value = constraints or {"Age": {"min": 0, "max": 120}}
        return client

    def test_finished_emitted_with_constraints(self, qtbot, qapp_instance, sample_df_clf):
        expected = {"sepal_length": {"min": 4.0, "max": 8.0}}
        client = self._make_client(constraints=expected)

        worker = LLMWorker(client, sample_df_clf, list(sample_df_clf.columns), "species")
        with qtbot.waitSignal(worker.finished, timeout=5000) as blocker:
            worker.start()
            worker.wait(5000)

        assert blocker.args[0] == expected

    def test_error_emitted_on_exception(self, qtbot, qapp_instance, sample_df_clf):
        client = self._make_client(raise_exc=ConnectionError("LM Studio not running"))

        worker = LLMWorker(client, sample_df_clf, list(sample_df_clf.columns), "species")
        with qtbot.waitSignal(worker.error, timeout=5000) as blocker:
            worker.start()
            worker.wait(5000)

        assert "ConnectionError" in blocker.args[0]
