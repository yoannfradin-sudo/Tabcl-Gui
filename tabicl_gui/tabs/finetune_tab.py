import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QCheckBox, QMessageBox,
    QProgressBar, QTextEdit, QRadioButton,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split

from tabicl_gui.utils import detect_task_type
from tabicl_gui.workers import FinetuneWorker


class FinetuneTab(QWidget):
    def __init__(self, state: dict, parent=None):
        super().__init__(parent)
        self.state = state
        self._worker = None
        self._loss_history: list[float] = []
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        notice = QLabel(
            "Le fine-tuning adapte les poids du modèle à votre dataset via PyTorch.\n"
            "Nécessite tabicl[finetune] et de préférence un GPU."
        )
        notice.setWordWrap(True)
        root.addWidget(notice)

        # ── Task type ──────────────────────────────────────────────────
        task_box = QGroupBox("Type de tâche")
        task_layout = QHBoxLayout(task_box)
        self._rb_clf = QRadioButton("Classification")
        self._rb_reg = QRadioButton("Régression")
        self._rb_clf.setChecked(True)
        self._rb_clf.toggled.connect(self._on_task_changed)
        task_layout.addWidget(self._rb_clf)
        task_layout.addWidget(self._rb_reg)
        task_layout.addStretch()
        root.addWidget(task_box)

        # ── Hyperparams ────────────────────────────────────────────────
        hp_box = QGroupBox("Hyperparamètres")
        hp_layout = QVBoxLayout(hp_box)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("epochs :"))
        self._epochs = QSpinBox()
        self._epochs.setRange(1, 500)
        self._epochs.setValue(50)
        row1.addWidget(self._epochs)

        row1.addWidget(QLabel("  patience :"))
        self._patience = QSpinBox()
        self._patience.setRange(1, 100)
        self._patience.setValue(10)
        row1.addWidget(self._patience)

        row1.addWidget(QLabel("  learning_rate :"))
        self._lr = QDoubleSpinBox()
        self._lr.setDecimals(7)
        self._lr.setRange(1e-7, 1e-1)
        self._lr.setSingleStep(1e-5)
        self._lr.setValue(1e-5)
        row1.addWidget(self._lr)
        row1.addStretch()
        hp_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("eval_metric :"))
        self._metric_combo = QComboBox()
        self._metric_combo.addItems(["roc_auc", "log_loss", "accuracy"])
        row2.addWidget(self._metric_combo)

        row2.addWidget(QLabel("  fraction validation :"))
        self._val_ratio = QDoubleSpinBox()
        self._val_ratio.setRange(0.05, 0.4)
        self._val_ratio.setSingleStep(0.05)
        self._val_ratio.setValue(0.2)
        row2.addWidget(self._val_ratio)

        self._early_stop_cb = QCheckBox("Early stopping")
        self._early_stop_cb.setChecked(True)
        row2.addWidget(self._early_stop_cb)
        row2.addStretch()
        hp_layout.addLayout(row2)

        root.addWidget(hp_box)

        # ── Controls ───────────────────────────────────────────────────
        ctrl_row = QHBoxLayout()
        self._btn_run = QPushButton("Lancer le fine-tuning")
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
        self._progress.setRange(0, 1)
        root.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(100)
        root.addWidget(self._log)

        # ── Loss chart ─────────────────────────────────────────────────
        self._fig = Figure(figsize=(6, 3), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        root.addWidget(self._canvas, 1)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _on_task_changed(self, checked: bool):
        if self._rb_clf.isChecked():
            self._metric_combo.clear()
            self._metric_combo.addItems(["roc_auc", "log_loss", "accuracy"])
        else:
            self._metric_combo.clear()
            self._metric_combo.addItems(["mse", "mae", "r2"])

    def on_data_confirmed(self, state: dict):
        self.state = state
        df = state["df"]
        target = state["target"]
        task = detect_task_type(df[target])
        if task == "classification":
            self._rb_clf.setChecked(True)
        else:
            self._rb_reg.setChecked(True)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run(self):
        if "df" not in self.state:
            QMessageBox.warning(self, "Pas de données", "Chargez d'abord des données.")
            return

        df = self.state["df"]
        target = self.state["target"]
        features = self.state["features"]
        task = "classification" if self._rb_clf.isChecked() else "regression"

        X = df[features].values
        y = df[target].values

        val_ratio = self._val_ratio.value()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_ratio, random_state=42
        )
        # Use last 20% of training as test
        X_tr, X_test, y_tr, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        params = {
            "epochs": self._epochs.value(),
            "learning_rate": self._lr.value(),
            "patience": self._patience.value(),
            "early_stopping": self._early_stop_cb.isChecked(),
            "eval_metric": self._metric_combo.currentText(),
            "verbose": True,
        }

        self._loss_history.clear()
        self._log.clear()
        self._progress.setRange(0, 0)
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)

        self._worker = FinetuneWorker(task, params, X_tr, y_tr, X_val, y_val, X_test)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _stop(self):
        if self._worker:
            self._worker.terminate()
        self._reset()
        self._log.append("Arrêté.")

    def _on_progress(self, msg: str):
        self._log.append(msg)
        # Try to parse a loss value from log lines like "epoch 5 loss=0.312"
        for token in msg.split():
            if "loss=" in token:
                try:
                    val = float(token.split("=")[1])
                    self._loss_history.append(val)
                    self._draw_loss()
                except ValueError:
                    pass

    def _on_done(self, model, preds, probas):
        self._reset()
        self.state["model"] = model
        self.state["predictions"] = preds
        self.state["probas"] = probas
        self.state["task"] = "classification" if self._rb_clf.isChecked() else "regression"
        self._log.append("Fine-tuning terminé — résultats disponibles dans l'onglet Résultats.")

    def _on_error(self, msg: str):
        self._reset()
        QMessageBox.critical(self, "Erreur fine-tuning", msg)

    def _reset(self):
        self._progress.setRange(0, 1)
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)

    def _draw_loss(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.plot(self._loss_history, color="steelblue")
        ax.set_xlabel("Étape")
        ax.set_ylabel("Loss")
        ax.set_title("Courbe de loss (fine-tuning)")
        self._canvas.draw()
