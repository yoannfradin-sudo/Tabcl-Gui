"""Onglet « Connaissances (LLM) ».

Utilise un endpoint compatible OpenAI (LM Studio en local) pour inférer
des contraintes métier par colonne, puis les rend éditables et les stocke
dans l'état partagé pour affiner l'entraînement et les prédictions.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QComboBox, QGroupBox, QTableWidget, QTableWidgetItem, QMessageBox,
    QProgressBar, QCheckBox, QRadioButton, QAbstractItemView, QTextEdit,
)

from tabicl_gui.llm import LLMClient, DEFAULT_BASE_URL, DEFAULT_API_KEY
from tabicl_gui.workers import LLMWorker


class KnowledgeTab(QWidget):
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

        notice = QLabel(
            "Le LLM local (LM Studio) agit comme expert métier : il infère des "
            "bornes physiques et contraintes par colonne (ex. Âge ∈ [0, 120]). "
            "Vous pouvez corriger ses propositions avant de les appliquer."
        )
        notice.setWordWrap(True)
        root.addWidget(notice)

        # ── Endpoint config ────────────────────────────────────────────
        cfg_box = QGroupBox("Connexion LM Studio (API compatible OpenAI)")
        cfg = QVBoxLayout(cfg_box)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Base URL :"))
        self._url_edit = QLineEdit(DEFAULT_BASE_URL)
        r1.addWidget(self._url_edit, 1)
        r1.addWidget(QLabel("Clé API :"))
        self._key_edit = QLineEdit(DEFAULT_API_KEY)
        self._key_edit.setMaximumWidth(140)
        r1.addWidget(self._key_edit)
        cfg.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Modèle :"))
        self._model_combo = QComboBox()
        self._model_combo.setEditable(True)
        self._model_combo.setMinimumWidth(280)
        r2.addWidget(self._model_combo, 1)
        self._btn_models = QPushButton("Lister les modèles")
        self._btn_models.clicked.connect(self._list_models)
        r2.addWidget(self._btn_models)
        cfg.addLayout(r2)

        root.addWidget(cfg_box)

        # ── Analyse ────────────────────────────────────────────────────
        an_row = QHBoxLayout()
        self._btn_analyze = QPushButton("Analyser les colonnes (LLM)")
        self._btn_analyze.setStyleSheet("font-weight: bold; padding: 6px;")
        self._btn_analyze.clicked.connect(self._analyze)
        an_row.addWidget(self._btn_analyze)
        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 1)
        an_row.addWidget(self._progress, 1)
        root.addLayout(an_row)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        root.addWidget(self._status)

        # ── Constraints table (editable) ───────────────────────────────
        tbl_box = QGroupBox("Contraintes métier (éditables)")
        tbl_layout = QVBoxLayout(tbl_box)
        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["Colonne", "Min", "Max", "Unité", "Justification"]
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        tbl_layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._btn_add = QPushButton("Ajouter une ligne")
        self._btn_add.clicked.connect(lambda: self._add_row())
        self._btn_del = QPushButton("Supprimer la ligne sélectionnée")
        self._btn_del.clicked.connect(self._del_row)
        btn_row.addWidget(self._btn_add)
        btn_row.addWidget(self._btn_del)
        btn_row.addStretch()
        tbl_layout.addLayout(btn_row)
        root.addWidget(tbl_box, 1)

        # ── Application options ─────────────────────────────────────────
        opt_box = QGroupBox("Application des contraintes")
        opt = QVBoxLayout(opt_box)

        self._apply_data_cb = QCheckBox(
            "Nettoyer les données d'entraînement selon les bornes"
        )
        self._apply_data_cb.setChecked(True)
        opt.addWidget(self._apply_data_cb)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode :"))
        self._rb_clip = QRadioButton("Borner (clip)")
        self._rb_drop = QRadioButton("Supprimer les lignes (drop)")
        self._rb_clip.setChecked(True)
        mode_row.addWidget(self._rb_clip)
        mode_row.addWidget(self._rb_drop)
        mode_row.addStretch()
        opt.addLayout(mode_row)

        self._apply_pred_cb = QCheckBox(
            "Borner les prédictions de régression selon la cible"
        )
        self._apply_pred_cb.setChecked(True)
        opt.addWidget(self._apply_pred_cb)

        self._btn_save = QPushButton("Valider les contraintes")
        self._btn_save.setStyleSheet("font-weight: bold; padding: 6px;")
        self._btn_save.clicked.connect(self._save)
        opt.addWidget(self._btn_save)

        root.addWidget(opt_box)

    # ------------------------------------------------------------------
    # Data lifecycle
    # ------------------------------------------------------------------

    def on_data_confirmed(self, state: dict):
        self.state = state

    # ------------------------------------------------------------------
    # Model listing
    # ------------------------------------------------------------------

    def _make_client(self):
        return LLMClient(
            base_url=self._url_edit.text().strip(),
            api_key=self._key_edit.text().strip(),
            model=self._model_combo.currentText().strip() or None,
        )

    def _list_models(self):
        try:
            models = self._make_client().list_models()
        except Exception as exc:
            QMessageBox.critical(self, "Connexion LM Studio", str(exc))
            return
        if not models:
            QMessageBox.information(self, "Modèles", "Aucun modèle chargé dans LM Studio.")
            return
        current = self._model_combo.currentText()
        self._model_combo.clear()
        self._model_combo.addItems(models)
        if current in models:
            self._model_combo.setCurrentText(current)

    # ------------------------------------------------------------------
    # Analyse
    # ------------------------------------------------------------------

    def _analyze(self):
        if "df" not in self.state:
            QMessageBox.warning(self, "Pas de données", "Chargez d'abord des données dans l'onglet Données.")
            return

        df = self.state["df"]
        features = self.state.get("features", list(df.columns))
        target = self.state.get("target")
        # On analyse features + cible (la cible sert au bornage des prédictions)
        columns = list(dict.fromkeys(features + ([target] if target else [])))

        self._progress.setRange(0, 0)
        self._btn_analyze.setEnabled(False)
        self._status.setText("Analyse en cours…")

        self._worker = LLMWorker(self._make_client(), df, columns, target)
        self._worker.progress.connect(self._status.setText)
        self._worker.finished.connect(self._on_analyzed)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_analyzed(self, constraints: dict):
        self._progress.setRange(0, 1)
        self._btn_analyze.setEnabled(True)
        self._populate_table(constraints)
        self._status.setText(
            f"{len(constraints)} contrainte(s) proposée(s). "
            "Vérifiez/corrigez puis cliquez « Valider »."
        )

    def _on_error(self, msg: str):
        self._progress.setRange(0, 1)
        self._btn_analyze.setEnabled(True)
        self._status.setText("Échec de l'analyse.")
        QMessageBox.critical(self, "Erreur LLM", msg)

    # ------------------------------------------------------------------
    # Table helpers
    # ------------------------------------------------------------------

    def _populate_table(self, constraints: dict):
        self._table.setRowCount(0)
        for name, c in constraints.items():
            self._add_row(name, c.get("min"), c.get("max"), c.get("unit"), c.get("rationale"))

    def _add_row(self, name="", mn=None, mx=None, unit=None, rationale=""):
        r = self._table.rowCount()
        self._table.insertRow(r)
        values = [
            name,
            "" if mn is None else str(mn),
            "" if mx is None else str(mx),
            unit or "",
            rationale or "",
        ]
        for c, v in enumerate(values):
            self._table.setItem(r, c, QTableWidgetItem(v))

    def _del_row(self):
        r = self._table.currentRow()
        if r >= 0:
            self._table.removeRow(r)

    def _collect_constraints(self) -> dict:
        out = {}
        for r in range(self._table.rowCount()):
            name_item = self._table.item(r, 0)
            if not name_item or not name_item.text().strip():
                continue
            name = name_item.text().strip()
            out[name] = {
                "min": _parse_num(self._table.item(r, 1)),
                "max": _parse_num(self._table.item(r, 2)),
                "unit": _cell_text(self._table.item(r, 3)),
                "rationale": _cell_text(self._table.item(r, 4)),
            }
        return out

    # ------------------------------------------------------------------
    # Save to state
    # ------------------------------------------------------------------

    def _save(self):
        constraints = self._collect_constraints()
        self.state["constraints"] = constraints
        self.state["constraints_apply_data"] = self._apply_data_cb.isChecked()
        self.state["constraints_apply_preds"] = self._apply_pred_cb.isChecked()
        self.state["constraints_mode"] = "drop" if self._rb_drop.isChecked() else "clip"
        QMessageBox.information(
            self, "Contraintes validées",
            f"{len(constraints)} contrainte(s) seront appliquées lors de l'entraînement.",
        )


def _parse_num(item):
    if item is None:
        return None
    txt = item.text().strip()
    if not txt:
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _cell_text(item):
    return item.text().strip() if item else ""
