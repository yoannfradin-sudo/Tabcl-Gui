"""Tests des onglets PyQt6.

• Aucun display réel requis (QT_QPA_PLATFORM=offscreen dans conftest).
• tabicl non installé — les workers sont mockés.
• Les dialogues (QMessageBox, QFileDialog) sont mockés.
"""
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tabicl_gui.tabs.data_tab import DataTab
from tabicl_gui.tabs.results_tab import ResultsTab
from tabicl_gui.tabs.train_tab import TrainTab


# ═══════════════════════════════════════════════════════════════════════
# DataTab
# ═══════════════════════════════════════════════════════════════════════

class TestDataTab:
    @pytest.fixture
    def tab(self, qapp_instance, state):
        t = DataTab(state)
        t.show()
        return t

    def test_load_csv_populates_table(self, qtbot, tab, tmp_csv, sample_df_clf):
        tab._load(tmp_csv, label=tmp_csv)
        # La table doit avoir au plus 200 lignes et le bon nombre de colonnes
        assert tab._table.rowCount() == min(len(sample_df_clf), 200)
        assert tab._table.columnCount() == len(sample_df_clf.columns)

    def test_target_combo_populated(self, qtbot, tab, tmp_csv, sample_df_clf):
        tab._load(tmp_csv, label=tmp_csv)
        assert tab._target_combo.count() == len(sample_df_clf.columns)

    def test_confirm_button_enabled_after_load(self, qtbot, tab, tmp_csv):
        assert not tab._btn_confirm.isEnabled()
        tab._load(tmp_csv, label=tmp_csv)
        assert tab._btn_confirm.isEnabled()

    def test_confirm_emits_signal_with_correct_keys(self, qtbot, tab, tmp_csv):
        tab._load(tmp_csv, label=tmp_csv)
        with qtbot.waitSignal(tab.data_confirmed, timeout=2000) as blocker:
            tab._confirm()
        state = blocker.args[0]
        assert "df" in state
        assert "target" in state
        assert "features" in state

    def test_confirm_target_not_in_features(self, qtbot, tab, tmp_csv, sample_df_clf):
        tab._load(tmp_csv, label=tmp_csv)
        with qtbot.waitSignal(tab.data_confirmed, timeout=2000) as blocker:
            tab._confirm()
        state = blocker.args[0]
        assert state["target"] not in state["features"]

    def test_confirm_without_features_shows_warning(self, qtbot, tab, tmp_csv):
        tab._load(tmp_csv, label=tmp_csv)
        tab._feat_list.clearSelection()
        with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warn:
            tab._confirm()
            mock_warn.assert_called_once()

    def test_invalid_path_shows_error(self, qtbot, tab):
        with patch("PyQt6.QtWidgets.QMessageBox.critical") as mock_err:
            tab._load("/nonexistent/path/file.csv", label="bad")
            mock_err.assert_called_once()

    def test_info_label_shows_shape(self, qtbot, tab, tmp_csv, sample_df_clf):
        tab._load(tmp_csv, label=tmp_csv)
        label_text = tab._info_label.text()
        assert str(sample_df_clf.shape[0]) in label_text
        assert str(sample_df_clf.shape[1]) in label_text

    def test_select_all_features_selects_all(self, qtbot, tab, tmp_csv, sample_df_clf):
        tab._load(tmp_csv, label=tmp_csv)
        tab._feat_list.clearSelection()
        tab._select_all_features()
        selected = [
            tab._feat_list.item(i).text()
            for i in range(tab._feat_list.count())
            if tab._feat_list.item(i).isSelected()
        ]
        assert len(selected) == tab._feat_list.count()


# ═══════════════════════════════════════════════════════════════════════
# TrainTab
# ═══════════════════════════════════════════════════════════════════════

class TestTrainTab:
    @pytest.fixture
    def tab(self, qapp_instance, state):
        t = TrainTab(state)
        t.show()
        return t

    @pytest.fixture
    def tab_with_data(self, tab, sample_df_clf, state):
        state.update({
            "df": sample_df_clf,
            "target": "species",
            "features": [c for c in sample_df_clf.columns if c != "species"],
        })
        tab.on_data_confirmed(state)
        return tab

    def test_run_without_data_shows_warning(self, qtbot, tab):
        with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warn:
            tab._run()
            mock_warn.assert_called_once()

    def test_on_data_confirmed_logs_target(self, qtbot, tab_with_data, state):
        log_text = tab_with_data._log.toPlainText()
        assert "species" in log_text

    def test_on_data_confirmed_logs_task(self, qtbot, tab_with_data):
        log_text = tab_with_data._log.toPlainText()
        assert "classification" in log_text.lower()

    def test_progress_bar_starts_indeterminate_on_run(self, qtbot, tab_with_data, state):
        fake_worker = MagicMock()
        fake_worker.progress = MagicMock()
        fake_worker.finished = MagicMock()
        fake_worker.error = MagicMock()
        # On remplace TrainWorker pour ne pas lancer de vraie inférence
        with patch("tabicl_gui.tabs.train_tab.TrainWorker", return_value=fake_worker):
            tab_with_data._run()
        # Progressbar en mode indéfini quand min == max == 0
        assert tab_with_data._progress.minimum() == 0
        assert tab_with_data._progress.maximum() == 0

    def test_on_finished_emits_training_done(self, qtbot, tab_with_data, state):
        # Simuler la fin du worker
        fake_model = MagicMock()
        fake_preds = np.array([0, 1, 0])
        # Pré-remplir X_test / y_test dans state
        state["X_test"] = np.zeros((3, 4))
        state["y_test"] = pd.Series([0, 1, 0])
        state["task"] = "classification"

        with qtbot.waitSignal(tab_with_data.training_done, timeout=2000):
            tab_with_data._on_finished(fake_model, fake_preds, None)

    def test_stop_resets_buttons(self, qtbot, tab_with_data):
        tab_with_data._btn_run.setEnabled(False)
        tab_with_data._btn_stop.setEnabled(True)
        tab_with_data._stop()
        assert tab_with_data._btn_run.isEnabled()
        assert not tab_with_data._btn_stop.isEnabled()


# ═══════════════════════════════════════════════════════════════════════
# ResultsTab
# ═══════════════════════════════════════════════════════════════════════

class TestResultsTab:
    @pytest.fixture
    def tab(self, qapp_instance, state):
        t = ResultsTab(state)
        t.show()
        return t

    def _clf_state(self, state, sample_df_clf):
        rng = np.random.default_rng(0)
        n = 20
        y_test = pd.Series(
            rng.choice(["setosa", "versicolor", "virginica"], n),
            name="species",
        )
        preds = y_test.values.copy()  # prédictions parfaites
        state.update({
            "task": "classification",
            "y_test": y_test,
            "predictions": preds,
            "probas": None,
            "model": MagicMock(),
            "X_test": np.zeros((n, 4)),
            "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        })
        return state

    def _reg_state(self, state, sample_df_reg):
        rng = np.random.default_rng(1)
        n = 20
        y_test = pd.Series(rng.standard_normal(n), name="price")
        preds = y_test.values + rng.standard_normal(n) * 0.1
        state.update({
            "task": "regression",
            "y_test": y_test,
            "predictions": preds,
            "probas": None,
            "model": MagicMock(),
            "X_test": np.zeros((n, 3)),
            "features": ["feature_a", "feature_b", "feature_c"],
        })
        return state

    def test_classification_metrics_displayed(self, qtbot, tab, state, sample_df_clf):
        s = self._clf_state(state, sample_df_clf)
        tab.on_training_done(s)
        label = tab._metrics_label.text()
        assert "Accuracy" in label
        assert "F1" in label

    def test_classification_pred_table_has_rows(self, qtbot, tab, state, sample_df_clf):
        s = self._clf_state(state, sample_df_clf)
        tab.on_training_done(s)
        assert tab._pred_table.rowCount() == 20

    def test_classification_confusion_matrix_drawn(self, qtbot, tab, state, sample_df_clf):
        s = self._clf_state(state, sample_df_clf)
        tab.on_training_done(s)
        # La figure doit contenir au moins un axe
        assert len(tab._fig.axes) > 0

    def test_regression_metrics_displayed(self, qtbot, tab, state, sample_df_reg):
        s = self._reg_state(state, sample_df_reg)
        tab.on_training_done(s)
        label = tab._metrics_label.text()
        assert "RMSE" in label
        assert "R²" in label

    def test_regression_scatter_drawn(self, qtbot, tab, state, sample_df_reg):
        s = self._reg_state(state, sample_df_reg)
        tab.on_training_done(s)
        assert len(tab._fig.axes) > 0

    def test_export_without_predictions_shows_warning(self, qtbot, tab, state):
        with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warn:
            tab._export()
            mock_warn.assert_called_once()

    def test_export_with_predictions_writes_csv(self, qtbot, tab, state, sample_df_clf, tmp_path):
        s = self._clf_state(state, sample_df_clf)
        tab.state = s
        csv_path = str(tmp_path / "preds.csv")
        with patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName", return_value=(csv_path, "")):
            with patch("PyQt6.QtWidgets.QMessageBox.information"):
                tab._export()
        assert os.path.isfile(csv_path)
        df = pd.read_csv(csv_path)
        assert "y_true" in df.columns
        assert "y_pred" in df.columns
        assert len(df) == 20

    def test_export_cancelled_does_nothing(self, qtbot, tab, state, sample_df_clf):
        s = self._clf_state(state, sample_df_clf)
        tab.state = s
        with patch("PyQt6.QtWidgets.QFileDialog.getSaveFileName", return_value=("", "")):
            tab._export()   # ne doit pas lever d'exception

    def test_shap_without_model_shows_warning(self, qtbot, tab):
        with patch("PyQt6.QtWidgets.QMessageBox.warning") as mock_warn:
            tab._run_shap()
            mock_warn.assert_called_once()
