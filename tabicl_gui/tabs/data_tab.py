import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QTableWidget, QTableWidgetItem, QComboBox, QListWidget,
    QListWidgetItem, QGroupBox, QSplitter, QMessageBox, QAbstractItemView,
)

from tabicl_gui.utils import load_data


class DataTab(QWidget):
    data_confirmed = pyqtSignal(object)   # emits the state dict

    def __init__(self, state: dict, parent=None):
        super().__init__(parent)
        self.state = state
        self._df = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ── Source row ────────────────────────────────────────────────
        src_box = QGroupBox("Source des données")
        src_layout = QVBoxLayout(src_box)

        file_row = QHBoxLayout()
        self._btn_open = QPushButton("Ouvrir fichier…")
        self._btn_open.clicked.connect(self._open_file)
        self._lbl_file = QLabel("Aucun fichier sélectionné")
        self._lbl_file.setWordWrap(True)
        file_row.addWidget(self._btn_open)
        file_row.addWidget(self._lbl_file, 1)
        src_layout.addLayout(file_row)

        url_row = QHBoxLayout()
        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText("https://... (CSV, Excel, JSON, Parquet)")
        self._btn_url = QPushButton("Charger URL")
        self._btn_url.clicked.connect(self._load_url)
        url_row.addWidget(QLabel("URL :"))
        url_row.addWidget(self._url_edit, 1)
        url_row.addWidget(self._btn_url)
        src_layout.addLayout(url_row)

        root.addWidget(src_box)

        # ── Main splitter: preview | column config ────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Preview pane
        preview_group = QGroupBox("Aperçu des données")
        prev_layout = QVBoxLayout(preview_group)
        self._info_label = QLabel("")
        prev_layout.addWidget(self._info_label)
        self._table = QTableWidget()
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        prev_layout.addWidget(self._table)
        splitter.addWidget(preview_group)

        # Column config pane
        col_group = QGroupBox("Configuration des colonnes")
        col_layout = QVBoxLayout(col_group)

        col_layout.addWidget(QLabel("Colonne cible (y) :"))
        self._target_combo = QComboBox()
        self._target_combo.currentTextChanged.connect(self._on_target_changed)
        col_layout.addWidget(self._target_combo)

        col_layout.addWidget(QLabel("Features utilisées (tout cocher = toutes) :"))
        self._feat_list = QListWidget()
        self._feat_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        col_layout.addWidget(self._feat_list, 1)

        btn_row = QHBoxLayout()
        btn_all = QPushButton("Tout")
        btn_all.clicked.connect(self._select_all_features)
        btn_none = QPushButton("Aucun")
        btn_none.clicked.connect(self._feat_list.clearSelection)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_none)
        col_layout.addLayout(btn_row)

        self._btn_confirm = QPushButton("Confirmer et passer à l'entraînement →")
        self._btn_confirm.setEnabled(False)
        self._btn_confirm.setStyleSheet("font-weight: bold; padding: 6px;")
        self._btn_confirm.clicked.connect(self._confirm)
        col_layout.addWidget(self._btn_confirm)

        splitter.addWidget(col_group)
        splitter.setSizes([600, 300])
        root.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Ouvrir un fichier de données",
            "",
            "Données tabulaires (*.csv *.xlsx *.xls *.json *.parquet);;Tous (*)",
        )
        if path:
            self._load(path, label=path)

    def _load_url(self):
        url = self._url_edit.text().strip()
        if not url:
            QMessageBox.warning(self, "URL vide", "Veuillez saisir une URL.")
            return
        self._load(url, label=url)

    def _load(self, src: str, label: str):
        try:
            df = load_data(src)
        except Exception as exc:
            QMessageBox.critical(self, "Erreur de chargement", str(exc))
            return

        self._df = df
        self._lbl_file.setText(label)
        self._populate_table(df)
        self._populate_columns(df)
        self._btn_confirm.setEnabled(True)

    def _populate_table(self, df: pd.DataFrame):
        preview = df.head(200)
        self._table.clear()
        self._table.setRowCount(len(preview))
        self._table.setColumnCount(len(preview.columns))
        self._table.setHorizontalHeaderLabels(list(preview.columns))

        for r, row in enumerate(preview.itertuples(index=False)):
            for c, val in enumerate(row):
                self._table.setItem(r, c, QTableWidgetItem(str(val)))

        self._table.resizeColumnsToContents()
        rows, cols = df.shape
        dtypes = ", ".join(f"{col}: {dtype}" for col, dtype in df.dtypes.items())
        missing = df.isnull().sum().sum()
        self._info_label.setText(
            f"{rows} lignes × {cols} colonnes  |  Valeurs manquantes : {missing}"
        )

    def _populate_columns(self, df: pd.DataFrame):
        self._target_combo.blockSignals(True)
        self._target_combo.clear()
        self._target_combo.addItems(list(df.columns))
        self._target_combo.setCurrentIndex(len(df.columns) - 1)
        self._target_combo.blockSignals(False)

        self._refresh_feature_list()

    def _refresh_feature_list(self):
        if self._df is None:
            return
        target = self._target_combo.currentText()
        self._feat_list.clear()
        for col in self._df.columns:
            if col == target:
                continue
            item = QListWidgetItem(col)
            self._feat_list.addItem(item)
            item.setSelected(True)

    def _on_target_changed(self, _text):
        self._refresh_feature_list()

    def _select_all_features(self):
        for i in range(self._feat_list.count()):
            self._feat_list.item(i).setSelected(True)

    def _confirm(self):
        if self._df is None:
            return

        target = self._target_combo.currentText()
        selected = [
            self._feat_list.item(i).text()
            for i in range(self._feat_list.count())
            if self._feat_list.item(i).isSelected()
        ]
        if not selected:
            QMessageBox.warning(self, "Aucune feature", "Sélectionnez au moins une feature.")
            return

        self.state["df"] = self._df
        self.state["target"] = target
        self.state["features"] = selected
        self.data_confirmed.emit(self.state)
