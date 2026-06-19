import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QSpinBox, QGroupBox, QMessageBox, QProgressBar, QTextEdit,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from tabicl_gui.workers import ForecastWorker


class ForecastTab(QWidget):
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
            "Utilisez cet onglet pour la prévision de séries temporelles univariées.\n"
            "Chargez d'abord vos données dans l'onglet Données (une colonne date + une colonne valeur)."
        )
        notice.setWordWrap(True)
        root.addWidget(notice)

        # ── Column config ──────────────────────────────────────────────
        col_box = QGroupBox("Configuration")
        col_layout = QHBoxLayout(col_box)

        col_layout.addWidget(QLabel("Colonne date/temps :"))
        self._date_combo = QComboBox()
        col_layout.addWidget(self._date_combo)

        col_layout.addWidget(QLabel("  Colonne valeur :"))
        self._value_combo = QComboBox()
        col_layout.addWidget(self._value_combo)

        col_layout.addWidget(QLabel("  ID série (optionnel) :"))
        self._id_combo = QComboBox()
        self._id_combo.addItem("(aucun)")
        col_layout.addWidget(self._id_combo)

        col_layout.addWidget(QLabel("  Horizon :"))
        self._horizon = QSpinBox()
        self._horizon.setRange(1, 1000)
        self._horizon.setValue(10)
        col_layout.addWidget(self._horizon)

        col_layout.addStretch()
        root.addWidget(col_box)

        btn_row = QHBoxLayout()
        self._btn_refresh = QPushButton("Actualiser les colonnes")
        self._btn_refresh.clicked.connect(self._refresh_columns)
        self._btn_run = QPushButton("Lancer la prévision")
        self._btn_run.setStyleSheet("font-weight: bold;")
        self._btn_run.clicked.connect(self._run)
        btn_row.addWidget(self._btn_refresh)
        btn_row.addWidget(self._btn_run)
        btn_row.addStretch()
        root.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.setRange(0, 1)
        root.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(80)
        root.addWidget(self._log)

        # ── Chart ──────────────────────────────────────────────────────
        self._fig = Figure(figsize=(7, 4), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        root.addWidget(self._canvas, 1)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _refresh_columns(self):
        if "df" not in self.state:
            QMessageBox.warning(self, "Pas de données", "Chargez d'abord des données.")
            return
        cols = list(self.state["df"].columns)
        for combo in (self._date_combo, self._value_combo):
            current = combo.currentText()
            combo.clear()
            combo.addItems(cols)
            if current in cols:
                combo.setCurrentText(current)

        self._id_combo.clear()
        self._id_combo.addItem("(aucun)")
        self._id_combo.addItems(cols)

    def on_data_confirmed(self, state: dict):
        self.state = state
        self._refresh_columns()

    def _run(self):
        if "df" not in self.state:
            QMessageBox.warning(self, "Pas de données", "Chargez d'abord des données.")
            return

        df = self.state["df"].copy()
        date_col = self._date_combo.currentText()
        val_col = self._value_combo.currentText()
        id_col = self._id_combo.currentText()
        horizon = self._horizon.value()

        try:
            df[date_col] = pd.to_datetime(df[date_col])
        except Exception as exc:
            QMessageBox.critical(self, "Erreur", f"Impossible de parser la colonne date : {exc}")
            return

        if id_col == "(aucun)":
            df["_item_id"] = "series_0"
            id_col = "_item_id"

        context_df = df[[id_col, date_col, val_col]].rename(
            columns={id_col: "item_id", date_col: "timestamp", val_col: "target"}
        )
        context_df = context_df.set_index(["item_id", "timestamp"])

        self._progress.setRange(0, 0)
        self._btn_run.setEnabled(False)

        self._worker = ForecastWorker(context_df, horizon)
        self._worker.progress.connect(self._log.append)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

        self._context_df = context_df
        self._date_col = date_col
        self._val_col = val_col

    def _on_done(self, pred_df):
        self._progress.setRange(0, 1)
        self._btn_run.setEnabled(True)
        self._draw_forecast(self._context_df, pred_df)

    def _on_error(self, msg: str):
        self._progress.setRange(0, 1)
        self._btn_run.setEnabled(True)
        QMessageBox.critical(self, "Erreur de prévision", msg)

    # ------------------------------------------------------------------
    # Chart
    # ------------------------------------------------------------------

    def _draw_forecast(self, context_df, pred_df):
        self._fig.clear()
        ax = self._fig.add_subplot(111)

        try:
            ctx = context_df.reset_index()
            ax.plot(ctx["timestamp"], ctx["target"], label="Historique", color="steelblue")
        except Exception:
            pass

        try:
            pred = pred_df.reset_index() if hasattr(pred_df, "reset_index") else pred_df
            ts_col = "timestamp" if "timestamp" in pred.columns else pred.columns[0]
            val_col = "mean" if "mean" in pred.columns else pred.columns[-1]
            ax.plot(pred[ts_col], pred[val_col], label="Prévision", color="tomato", linestyle="--")
        except Exception:
            pass

        ax.set_title("Prévision de série temporelle")
        ax.legend()
        ax.tick_params(axis="x", rotation=30)
        self._canvas.draw()
