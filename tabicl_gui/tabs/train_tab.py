import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QGroupBox, QRadioButton,
    QTextEdit, QProgressBar, QMessageBox, QButtonGroup, QSizePolicy,
    QLineEdit, QFileDialog,
)
from sklearn.model_selection import train_test_split

from tabicl_gui.utils import detect_task_type
from tabicl_gui.workers import TrainWorker


class TrainTab(QWidget):
    training_done = pyqtSignal(object)    # emits updated state

    def __init__(self, state: dict, parent=None):
        super().__init__(parent)
        self.state = state
        self._worker = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ── Task type ─────────────────────────────────────────────────
        task_box = QGroupBox("Type de tâche")
        task_layout = QHBoxLayout(task_box)
        self._rb_clf = QRadioButton("Classification")
        self._rb_reg = QRadioButton("Régression")
        self._rb_clf.setChecked(True)
        task_layout.addWidget(self._rb_clf)
        task_layout.addWidget(self._rb_reg)
        task_layout.addStretch()
        root.addWidget(task_box)

        # ── Split ─────────────────────────────────────────────────────
        split_box = QGroupBox("Division train / test")
        split_layout = QHBoxLayout(split_box)
        split_layout.addWidget(QLabel("Ratio test :"))
        self._test_ratio = QDoubleSpinBox()
        self._test_ratio.setRange(0.05, 0.5)
        self._test_ratio.setSingleStep(0.05)
        self._test_ratio.setValue(0.2)
        split_layout.addWidget(self._test_ratio)
        self._stratify_cb = QCheckBox("Stratification")
        self._stratify_cb.setChecked(True)
        split_layout.addWidget(self._stratify_cb)
        split_layout.addStretch()
        root.addWidget(split_box)

        # ── TabICL params ──────────────────────────────────────────────
        params_box = QGroupBox("Paramètres TabICL")
        grid = QVBoxLayout(params_box)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("n_estimators :"))
        self._n_est = QSpinBox()
        self._n_est.setRange(1, 50)
        self._n_est.setValue(8)
        row1.addWidget(self._n_est)

        row1.addWidget(QLabel("  device :"))
        self._device = QComboBox()
        self._device.addItems(["auto", "cpu", "cuda", "mps"])
        row1.addWidget(self._device)

        row1.addWidget(QLabel("  random_state :"))
        self._seed = QSpinBox()
        self._seed.setRange(0, 99999)
        self._seed.setValue(42)
        row1.addWidget(self._seed)
        row1.addStretch()
        grid.addLayout(row1)

        row2 = QHBoxLayout()
        self._kvcache_cb = QCheckBox("kv_cache")
        self._verbose_cb = QCheckBox("verbose")
        row2.addWidget(self._kvcache_cb)
        row2.addWidget(self._verbose_cb)
        row2.addStretch()
        grid.addLayout(row2)

        # Checkpoint local (utile en environnement hors ligne)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Checkpoint local :"))
        self._ckpt_edit = QLineEdit()
        self._ckpt_edit.setPlaceholderText("(vide = téléchargement auto / cache)")
        self._btn_ckpt = QPushButton("Parcourir…")
        self._btn_ckpt.clicked.connect(self._pick_checkpoint)
        row3.addWidget(self._ckpt_edit, 1)
        row3.addWidget(self._btn_ckpt)
        grid.addLayout(row3)

        root.addWidget(params_box)

        # ── Run controls ───────────────────────────────────────────────
        ctrl_row = QHBoxLayout()
        self._btn_run = QPushButton("Lancer l'entraînement")
        self._btn_run.setStyleSheet("font-weight: bold; padding: 6px;")
        self._btn_run.clicked.connect(self._run)
        self._btn_stop = QPushButton("Arrêter")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop)
        ctrl_row.addWidget(self._btn_run)
        ctrl_row.addWidget(self._btn_stop)
        ctrl_row.addStretch()
        root.addLayout(ctrl_row)

        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        root.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(180)
        root.addWidget(self._log)

        root.addStretch()

    # ------------------------------------------------------------------
    # Populate from data tab
    # ------------------------------------------------------------------

    def on_data_confirmed(self, state: dict):
        df = state["df"]
        target = state["target"]
        task = detect_task_type(df[target])
        if task == "classification":
            self._rb_clf.setChecked(True)
        else:
            self._rb_reg.setChecked(True)
        self._log.append(
            f"Données chargées : {df.shape[0]} lignes, "
            f"{len(state['features'])} features, cible '{target}' "
            f"(tâche détectée : {task})"
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _run(self):
        if "df" not in self.state:
            QMessageBox.warning(self, "Pas de données", "Chargez d'abord des données dans l'onglet Données.")
            return

        df = self.state["df"]
        target = self.state["target"]
        features = self.state["features"]
        task = "classification" if self._rb_clf.isChecked() else "regression"

        X = df[features]
        y = df[target]

        ratio = self._test_ratio.value()
        stratify = y if (task == "classification" and self._stratify_cb.isChecked()) else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=ratio, random_state=self._seed.value(), stratify=stratify
            )
        except ValueError as exc:
            QMessageBox.critical(self, "Erreur split", str(exc))
            return

        device_val = self._device.currentText()
        params = {
            "n_estimators": self._n_est.value(),
            "random_state": self._seed.value(),
            "kv_cache": self._kvcache_cb.isChecked(),
            "verbose": self._verbose_cb.isChecked(),
        }
        if device_val != "auto":
            params["device"] = device_val

        ckpt = self._ckpt_edit.text().strip()
        if ckpt:
            params["model_path"] = ckpt
            params["allow_auto_download"] = False

        self.state.update({
            "task": task,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
        })

        self._log.clear()
        self._log.append(
            f"Train : {len(X_train)} | Test : {len(X_test)} | Task : {task}"
        )
        self._progress.setRange(0, 0)
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)

        self._worker = TrainWorker(task, params, X_train.values, y_train.values, X_test.values)
        self._worker.progress.connect(self._log.append)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _pick_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Sélectionner un checkpoint TabICL", "", "Checkpoint (*.ckpt);;Tous (*)"
        )
        if path:
            self._ckpt_edit.setText(path)

    def _stop(self):
        if self._worker:
            self._worker.stop()
            self._worker.quit()
        self._reset_progress()
        self._log.append("Arrêté par l'utilisateur.")

    def _on_finished(self, model, preds, probas):
        self._reset_progress()
        self.state["model"] = model
        self.state["predictions"] = preds
        self.state["probas"] = probas
        self._log.append("Entraînement terminé — résultats disponibles dans l'onglet Résultats.")
        self.training_done.emit(self.state)

    def _on_error(self, msg: str):
        self._reset_progress()
        QMessageBox.critical(self, "Erreur d'entraînement", msg)
        self._log.append(f"ERREUR:\n{msg}")

    def _reset_progress(self):
        self._progress.setRange(0, 1)
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
