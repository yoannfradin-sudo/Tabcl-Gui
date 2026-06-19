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
from tabicl_gui.workers import CVWorker, TrainWorker


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

        # ── Split / CV ────────────────────────────────────────────────
        split_box = QGroupBox("Division train / test")
        split_layout = QVBoxLayout(split_box)

        row_split = QHBoxLayout()
        row_split.addWidget(QLabel("Ratio test :"))
        self._test_ratio = QDoubleSpinBox()
        self._test_ratio.setRange(0.05, 0.5)
        self._test_ratio.setSingleStep(0.05)
        self._test_ratio.setValue(0.2)
        row_split.addWidget(self._test_ratio)
        self._stratify_cb = QCheckBox("Stratification")
        self._stratify_cb.setChecked(True)
        row_split.addWidget(self._stratify_cb)
        row_split.addStretch()
        split_layout.addLayout(row_split)

        row_cv = QHBoxLayout()
        self._cv_cb = QCheckBox("Validation croisée k-fold")
        self._cv_cb.toggled.connect(self._on_cv_toggled)
        row_cv.addWidget(self._cv_cb)
        row_cv.addWidget(QLabel("  k :"))
        self._k_spin = QSpinBox()
        self._k_spin.setRange(2, 20)
        self._k_spin.setValue(5)
        self._k_spin.setEnabled(False)
        row_cv.addWidget(self._k_spin)
        self._cv_warn = QLabel("⚠ Attention : k × plus lent")
        self._cv_warn.setStyleSheet("color: #b05000; font-size: 11px;")
        self._cv_warn.setVisible(False)
        row_cv.addWidget(self._cv_warn)
        row_cv.addStretch()
        split_layout.addLayout(row_cv)

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

        # Injection des connaissances métier (LLM) : nettoyage des données
        constraints = self.state.get("constraints")
        if constraints and self.state.get("constraints_apply_data"):
            from tabicl_gui.llm import apply_constraints_to_data
            mode = self.state.get("constraints_mode", "clip")
            subset = df[features + [target]]
            subset, report = apply_constraints_to_data(subset, constraints, mode)
            for line in report:
                self._log.append(f"[LLM] {line}")
            if not report:
                self._log.append("[LLM] Aucune valeur hors bornes détectée.")
            X = subset[features]
            y = subset[target]
        else:
            X = df[features]
            y = df[target]

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

        self._log.clear()
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)

        if self._cv_cb.isChecked():
            k = self._k_spin.value()
            stratified = task == "classification"
            self.state.update({"task": task, "cv_mode": True, "cv_k": k})
            self._log.append(
                f"Cross-validation {k}-fold | Task : {task} | {len(X)} échantillons"
            )
            self._progress.setRange(0, k)
            self._progress.setValue(0)
            self._worker = CVWorker(task, params, X.values, y.values, k=k, stratified=stratified)
            self._worker.progress.connect(self._log.append)
            self._worker.fold_done.connect(lambda fi, _k: self._progress.setValue(fi))
            self._worker.finished.connect(self._on_cv_finished)
            self._worker.error.connect(self._on_error)
            self._worker.start()
        else:
            ratio = self._test_ratio.value()
            stratify = y if (task == "classification" and self._stratify_cb.isChecked()) else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=ratio, random_state=self._seed.value(), stratify=stratify
                )
            except ValueError as exc:
                QMessageBox.critical(self, "Erreur split", str(exc))
                self._reset_progress()
                return

            self.state.update({
                "task": task, "cv_mode": False,
                "X_train": X_train, "X_test": X_test,
                "y_train": y_train, "y_test": y_test,
            })
            self._log.append(
                f"Train : {len(X_train)} | Test : {len(X_test)} | Task : {task}"
            )
            self._progress.setRange(0, 0)
            self._worker = TrainWorker(task, params, X_train.values, y_train.values, X_test.values)
            self._worker.progress.connect(self._log.append)
            self._worker.finished.connect(self._on_finished)
            self._worker.error.connect(self._on_error)
            self._worker.start()

    def _on_cv_toggled(self, checked: bool):
        self._k_spin.setEnabled(checked)
        self._cv_warn.setVisible(checked)
        self._test_ratio.setEnabled(not checked)
        self._stratify_cb.setEnabled(not checked)

    def _on_cv_finished(self, all_preds, all_true, last_model):
        self._reset_progress()
        k = self.state.get("cv_k", self._k_spin.value())

        # Injection LLM : bornage des prédictions (régression)
        task = self.state.get("task")
        constraints = self.state.get("constraints")
        if (
            task == "regression"
            and constraints
            and self.state.get("constraints_apply_preds")
        ):
            target = self.state.get("target")
            tc = constraints.get(target)
            if tc and (tc.get("min") is not None or tc.get("max") is not None):
                from tabicl_gui.llm import clip_predictions
                all_preds = clip_predictions(all_preds, tc)
                self._log.append(
                    f"[LLM] Prédictions bornées à [{tc.get('min')}, {tc.get('max')}]."
                )

        # Cast from object dtype (used in CVWorker) to concrete dtype for sklearn
        try:
            all_preds = np.array(all_preds.tolist())
            all_true = np.array(all_true.tolist())
        except Exception:
            pass

        self.state["model"] = last_model
        self.state["predictions"] = all_preds
        self.state["probas"] = None
        self.state["cv_preds"] = all_preds
        self.state["cv_true"] = all_true
        self._log.append(
            f"Cross-validation terminée ({k} folds) — résultats disponibles dans l'onglet Résultats."
        )
        self.training_done.emit(self.state)

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

        # Injection des connaissances métier (LLM) : bornage des prédictions
        task = self.state.get("task")
        constraints = self.state.get("constraints")
        if (
            task == "regression"
            and constraints
            and self.state.get("constraints_apply_preds")
        ):
            target = self.state.get("target")
            tc = constraints.get(target)
            if tc and (tc.get("min") is not None or tc.get("max") is not None):
                from tabicl_gui.llm import clip_predictions
                preds = clip_predictions(preds, tc)
                self._log.append(
                    f"[LLM] Prédictions bornées à "
                    f"[{tc.get('min')}, {tc.get('max')}] (cible '{target}')."
                )

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
