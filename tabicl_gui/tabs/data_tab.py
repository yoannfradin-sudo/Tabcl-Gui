import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QTableWidget, QTableWidgetItem, QComboBox, QListWidget,
    QListWidgetItem, QGroupBox, QSplitter, QMessageBox, QAbstractItemView,
    QTabWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

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

        # Left pane: sub-tabs (Aperçu / Exploration)
        left_tabs = QTabWidget()

        # ── Sub-tab 1 : Aperçu ────────────────────────────────────────
        preview_widget = QWidget()
        prev_layout = QVBoxLayout(preview_widget)
        self._info_label = QLabel("")
        prev_layout.addWidget(self._info_label)
        self._table = QTableWidget()
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        prev_layout.addWidget(self._table)
        left_tabs.addTab(preview_widget, "Aperçu")

        # ── Sub-tab 2 : Exploration ───────────────────────────────────
        eda_widget = QWidget()
        eda_layout = QVBoxLayout(eda_widget)

        eda_ctrl = QHBoxLayout()
        eda_ctrl.addWidget(QLabel("Colonne :"))
        self._eda_col_combo = QComboBox()
        self._eda_col_combo.currentTextChanged.connect(self._draw_column_chart)
        eda_ctrl.addWidget(self._eda_col_combo, 1)
        self._btn_corr   = QPushButton("Corrélation")
        self._btn_corr.clicked.connect(self._draw_correlation)
        self._btn_missing = QPushButton("Valeurs manquantes")
        self._btn_missing.clicked.connect(self._draw_missing)
        eda_ctrl.addWidget(self._btn_corr)
        eda_ctrl.addWidget(self._btn_missing)
        eda_layout.addLayout(eda_ctrl)

        self._eda_fig    = Figure(figsize=(6, 3), tight_layout=True)
        self._eda_canvas = FigureCanvasQTAgg(self._eda_fig)
        eda_layout.addWidget(self._eda_canvas, 1)
        left_tabs.addTab(eda_widget, "Exploration")

        splitter.addWidget(left_tabs)

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

        self._eda_col_combo.blockSignals(True)
        self._eda_col_combo.clear()
        self._eda_col_combo.addItems(list(df.columns))
        self._eda_col_combo.blockSignals(False)

        self._refresh_feature_list()
        if len(df.columns):
            self._draw_column_chart(df.columns[0])

    # ------------------------------------------------------------------
    # EDA charts
    # ------------------------------------------------------------------

    def _draw_column_chart(self, col: str):
        if self._df is None or not col or col not in self._df.columns:
            return
        s = self._df[col].dropna()
        self._eda_fig.clear()
        ax = self._eda_fig.add_subplot(111)

        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 5:
            # Histogram + outlier markers
            ax.hist(s, bins=30, color="#4C72B0", edgecolor="white", alpha=0.8, label="Valeurs")
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = s[(s < lo) | (s > hi)]
            if not outliers.empty:
                ax.axvline(lo, color="red", ls="--", lw=1.2, label=f"Bornes IQR ({len(outliers)} outliers)")
                ax.axvline(hi, color="red", ls="--", lw=1.2)
            ax.set_xlabel(col)
            ax.set_ylabel("Effectif")
            ax.set_title(f"Distribution de '{col}'")
            if not outliers.empty:
                ax.legend(fontsize=8)
        else:
            # Bar chart of value counts
            vc = self._df[col].value_counts().head(20)
            ax.barh(vc.index.astype(str)[::-1], vc.values[::-1], color="#4C72B0")
            ax.set_xlabel("Effectif")
            ax.set_title(f"Fréquences de '{col}'")

        n_nan = int(self._df[col].isnull().sum())
        if n_nan:
            ax.set_title(ax.get_title() + f"  [{n_nan} NaN]")

        self._eda_canvas.draw()

    def _draw_correlation(self):
        if self._df is None:
            return
        num_df = self._df.select_dtypes(include="number")
        if num_df.shape[1] < 2:
            QMessageBox.information(self, "Corrélation", "Pas assez de colonnes numériques.")
            return
        corr = num_df.corr()
        self._eda_fig.clear()
        ax = self._eda_fig.add_subplot(111)
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
        self._eda_fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(corr.columns, fontsize=7)
        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if abs(corr.values[i, j]) > 0.5 else "black")
        ax.set_title("Matrice de corrélation")
        self._eda_canvas.draw()

    def _draw_missing(self):
        if self._df is None:
            return
        miss = self._df.isnull().sum()
        miss = miss[miss > 0].sort_values()
        if miss.empty:
            QMessageBox.information(self, "Valeurs manquantes", "Aucune valeur manquante.")
            return
        self._eda_fig.clear()
        ax = self._eda_fig.add_subplot(111)
        pct = miss / len(self._df) * 100
        bars = ax.barh(miss.index.astype(str), pct.values, color="#DD8452")
        ax.set_xlabel("% manquant")
        ax.set_title("Valeurs manquantes par colonne")
        for bar, v in zip(bars, miss.values):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{v}", va="center", fontsize=8)
        self._eda_canvas.draw()

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
