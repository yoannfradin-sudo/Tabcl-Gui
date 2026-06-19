import base64
import io
import numpy as np
import pandas as pd
from datetime import datetime
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QGroupBox, QSplitter, QFileDialog, QMessageBox,
    QTextEdit, QProgressBar, QAbstractItemView,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from tabicl_gui.utils import compute_classification_metrics, compute_regression_metrics
from tabicl_gui.workers import ShapWorker


class ResultsTab(QWidget):
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

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: metrics + SHAP ──────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)

        metrics_box = QGroupBox("Métriques")
        self._metrics_layout = QVBoxLayout(metrics_box)
        self._metrics_label = QLabel("(aucun résultat)")
        self._metrics_label.setWordWrap(True)
        self._metrics_layout.addWidget(self._metrics_label)
        left_layout.addWidget(metrics_box)

        report_box = QGroupBox("Rapport détaillé")
        report_layout = QVBoxLayout(report_box)
        self._report_text = QTextEdit()
        self._report_text.setReadOnly(True)
        self._report_text.setFont(self._report_text.font())
        self._report_text.setMaximumHeight(140)
        report_layout.addWidget(self._report_text)
        left_layout.addWidget(report_box)

        shap_box = QGroupBox("Explicabilité SHAP")
        shap_layout = QVBoxLayout(shap_box)
        self._btn_shap = QPushButton("Calculer les valeurs SHAP")
        self._btn_shap.clicked.connect(self._run_shap)
        self._shap_progress = QProgressBar()
        self._shap_progress.setTextVisible(False)
        self._shap_progress.setRange(0, 1)
        shap_layout.addWidget(self._btn_shap)
        shap_layout.addWidget(self._shap_progress)
        self._shap_fig = Figure(figsize=(4, 3), tight_layout=True)
        self._shap_canvas = FigureCanvasQTAgg(self._shap_fig)
        shap_layout.addWidget(self._shap_canvas)
        left_layout.addWidget(shap_box, 1)

        splitter.addWidget(left)

        # ── Right: chart + predictions table ──────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)

        chart_box = QGroupBox("Visualisation")
        chart_layout = QVBoxLayout(chart_box)
        self._fig = Figure(figsize=(5, 4), tight_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        chart_layout.addWidget(self._canvas)
        right_layout.addWidget(chart_box, 1)

        pred_box = QGroupBox("Prédictions")
        pred_layout = QVBoxLayout(pred_box)
        self._pred_table = QTableWidget()
        self._pred_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._pred_table.setAlternatingRowColors(True)
        pred_layout.addWidget(self._pred_table)

        btn_row = QHBoxLayout()
        self._btn_export = QPushButton("Exporter les prédictions (CSV)…")
        self._btn_export.clicked.connect(self._export)
        btn_row.addWidget(self._btn_export)
        self._btn_export_html = QPushButton("Générer rapport HTML…")
        self._btn_export_html.clicked.connect(self._export_html)
        btn_row.addWidget(self._btn_export_html)
        pred_layout.addLayout(btn_row)

        right_layout.addWidget(pred_box, 1)
        splitter.addWidget(right)

        splitter.setSizes([400, 600])
        root.addWidget(splitter)

    # ------------------------------------------------------------------
    # Called when training completes
    # ------------------------------------------------------------------

    def on_training_done(self, state: dict):
        self.state = state
        task = state.get("task", "classification")
        cv_mode = state.get("cv_mode", False)
        k = state.get("cv_k", 5)

        if cv_mode:
            y_vals = state["cv_true"]
            preds = state["cv_preds"]
            probas = None
        else:
            y_vals = state["y_test"].values
            preds = state["predictions"]
            probas = state.get("probas")

        suffix = f"\n(cross-validé, {k} folds)" if cv_mode else ""

        if task == "classification":
            m = compute_classification_metrics(y_vals, preds, probas)
            parts = [
                f"Accuracy : {m['accuracy']:.4f}{suffix}",
                f"F1 : {m['f1']:.4f}",
            ]
            if "roc_auc" in m:
                parts.append(f"ROC-AUC : {m['roc_auc']:.4f}")
            self._metrics_label.setText("\n".join(parts))
            self._report_text.setPlainText(m.get("report", ""))
            self._draw_confusion(m["confusion_matrix"], m["classes"])
        else:
            m = compute_regression_metrics(y_vals, preds)
            parts = [
                f"RMSE : {m['rmse']:.4f}{suffix}",
                f"MAE  : {m['mae']:.4f}",
                f"R²   : {m['r2']:.4f}",
            ]
            self._metrics_label.setText("\n".join(parts))
            self._report_text.clear()
            self._draw_scatter(y_vals, preds)

        self._populate_pred_table(state, preds, probas, task, cv_mode)

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def _draw_confusion(self, cm, classes):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        self._fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title("Matrice de confusion")
        self._canvas.draw()

    def _draw_scatter(self, y_true, y_pred):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.scatter(y_true, y_pred, alpha=0.5, s=20)
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], "r--", lw=1)
        ax.set_xlabel("Y réel")
        ax.set_ylabel("Y prédit")
        ax.set_title("Réel vs Prédit")
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Predictions table
    # ------------------------------------------------------------------

    def _populate_pred_table(self, state, preds, probas, task, cv_mode=False):
        if cv_mode:
            y_vals = state["cv_true"]
            idx = list(range(len(y_vals)))
        else:
            y_test = state["y_test"]
            idx = y_test.index.tolist()
            y_vals = y_test.values

        has_proba = probas is not None and task == "classification"

        cols = ["Index", "Y réel", "Y prédit"]
        if has_proba and probas.ndim == 2:
            for k in range(probas.shape[1]):
                cols.append(f"Proba_{k}")

        self._pred_table.clear()
        self._pred_table.setRowCount(len(preds))
        self._pred_table.setColumnCount(len(cols))
        self._pred_table.setHorizontalHeaderLabels(cols)

        for r in range(len(preds)):
            items = [str(idx[r]), str(y_vals[r]), str(preds[r])]
            if has_proba and probas.ndim == 2:
                for k in range(probas.shape[1]):
                    items.append(f"{probas[r, k]:.4f}")
            for c, val in enumerate(items):
                self._pred_table.setItem(r, c, QTableWidgetItem(val))

        self._pred_table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export(self):
        if "predictions" not in self.state:
            QMessageBox.warning(self, "Pas de prédictions", "Lancez d'abord un entraînement.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer les prédictions", "predictions.csv", "CSV (*.csv)")
        if not path:
            return
        preds = self.state["predictions"]
        probas = self.state.get("probas")
        task = self.state.get("task", "classification")
        cv_mode = self.state.get("cv_mode", False)

        if cv_mode:
            y_vals = self.state["cv_true"]
            df = pd.DataFrame({"index": range(len(y_vals)), "y_true": y_vals, "y_pred": preds})
        else:
            y_test = self.state["y_test"]
            df = pd.DataFrame({"index": y_test.index, "y_true": y_test.values, "y_pred": preds})
            if probas is not None and task == "classification" and probas.ndim == 2:
                for k in range(probas.shape[1]):
                    df[f"proba_{k}"] = probas[:, k]
        df.to_csv(path, index=False)
        QMessageBox.information(self, "Exporté", f"Prédictions sauvegardées dans :\n{path}")

    def _export_html(self):
        if "predictions" not in self.state:
            QMessageBox.warning(self, "Pas de prédictions", "Lancez d'abord un entraînement.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer le rapport HTML", "rapport_tabicl.html", "HTML (*.html)"
        )
        if not path:
            return

        task = self.state.get("task", "classification")
        cv_mode = self.state.get("cv_mode", False)
        k = self.state.get("cv_k", 5)
        features = self.state.get("features", [])
        target = self.state.get("target", "")

        if cv_mode:
            y_vals = self.state["cv_true"]
            preds = self.state["cv_preds"]
            probas = None
        else:
            y_vals = self.state["y_test"].values
            preds = self.state["predictions"]
            probas = self.state.get("probas")

        # Encode main figure
        buf = io.BytesIO()
        self._fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        fig_b64 = base64.b64encode(buf.getvalue()).decode()

        # Encode SHAP figure if axes present
        shap_b64 = None
        try:
            if self._shap_fig.get_axes():
                buf2 = io.BytesIO()
                self._shap_fig.savefig(buf2, format="png", dpi=100, bbox_inches="tight")
                shap_b64 = base64.b64encode(buf2.getvalue()).decode()
        except Exception:
            pass

        # Compute metrics
        if task == "classification":
            m = compute_classification_metrics(y_vals, preds, probas)
            metrics_rows = [("Accuracy", f"{m['accuracy']:.4f}"), ("F1 (weighted)", f"{m['f1']:.4f}")]
            if "roc_auc" in m:
                metrics_rows.append(("ROC-AUC", f"{m['roc_auc']:.4f}"))
        else:
            m = compute_regression_metrics(y_vals, preds)
            metrics_rows = [
                ("RMSE", f"{m['rmse']:.4f}"), ("MAE", f"{m['mae']:.4f}"),
                ("R²", f"{m['r2']:.4f}"), ("MSE", f"{m['mse']:.4f}"),
            ]

        mode_str = f"Cross-validation ({k} folds)" if cv_mode else "Train / Test split"
        feat_preview = ", ".join(features[:10]) + ("…" if len(features) > 10 else "")
        rows_html = "\n".join(f"  <tr><td>{n}</td><td><b>{v}</b></td></tr>" for n, v in metrics_rows)
        n_show = min(50, len(preds))
        pred_rows = "\n".join(
            f"  <tr><td>{i}</td><td>{y_vals[i]}</td><td>{preds[i]}</td></tr>"
            for i in range(n_show)
        )
        shap_section = (
            f'<h2>Explicabilité SHAP</h2>'
            f'<img src="data:image/png;base64,{shap_b64}" style="max-width:100%;"/>'
            if shap_b64 else ""
        )

        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Rapport TabICL — {datetime.now():%Y-%m-%d %H:%M}</title>
<style>
  body {{ font-family: sans-serif; margin: 40px; color: #333; }}
  h1 {{ color: #2c5f8a; }} h2 {{ color: #4a4a4a; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; margin: 12px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 12px; text-align: left; }}
  th {{ background: #f0f0f0; }} tr:nth-child(even) {{ background: #f9f9f9; }}
  img {{ border: 1px solid #ddd; border-radius: 4px; margin: 8px 0; }}
</style>
</head>
<body>
<h1>Rapport TabICL</h1>
<p>Généré le {datetime.now():%Y-%m-%d à %H:%M}</p>
<h2>Paramètres</h2>
<table>
  <tr><th>Paramètre</th><th>Valeur</th></tr>
  <tr><td>Tâche</td><td>{task}</td></tr>
  <tr><td>Cible</td><td>{target}</td></tr>
  <tr><td>Features ({len(features)})</td><td>{feat_preview}</td></tr>
  <tr><td>Mode</td><td>{mode_str}</td></tr>
</table>
<h2>Métriques</h2>
<table>
  <tr><th>Métrique</th><th>Valeur</th></tr>
{rows_html}
</table>
<h2>Visualisation</h2>
<img src="data:image/png;base64,{fig_b64}" style="max-width:100%;"/>
{shap_section}
<h2>Prédictions (50 premières)</h2>
<table>
  <tr><th>#</th><th>Y réel</th><th>Y prédit</th></tr>
{pred_rows}
</table>
</body>
</html>"""

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(html)
        QMessageBox.information(self, "Exporté", f"Rapport HTML sauvegardé dans :\n{path}")

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    def _run_shap(self):
        if "model" not in self.state:
            QMessageBox.warning(self, "Pas de modèle", "Lancez d'abord un entraînement.")
            return
        if "X_test" not in self.state:
            QMessageBox.warning(
                self, "Mode cross-validation",
                "SHAP n'est pas disponible en mode cross-validation (pas de jeu de test isolé).",
            )
            return
        self._shap_progress.setRange(0, 0)
        self._btn_shap.setEnabled(False)
        X_test = self.state["X_test"]
        features = self.state["features"]
        self._worker = ShapWorker(self.state["model"], X_test, features)
        self._worker.progress.connect(lambda m: None)
        self._worker.finished.connect(self._on_shap_done)
        self._worker.error.connect(self._on_shap_error)
        self._worker.start()

    def _on_shap_done(self, shap_vals, feature_names):
        self._shap_progress.setRange(0, 1)
        self._btn_shap.setEnabled(True)
        try:
            import shap
            self._shap_fig.clear()
            ax = self._shap_fig.add_subplot(111)
            if hasattr(shap_vals, "values"):
                vals = shap_vals.values
            else:
                vals = shap_vals
            if vals.ndim == 3:
                vals = vals[:, :, 1]
            mean_abs = np.abs(vals).mean(axis=0)
            sorted_idx = np.argsort(mean_abs)[-15:]
            ax.barh(
                [feature_names[i] for i in sorted_idx],
                mean_abs[sorted_idx],
            )
            ax.set_xlabel("|SHAP| moyen")
            ax.set_title("Importance des features (SHAP)")
            self._shap_canvas.draw()
        except Exception as exc:
            QMessageBox.warning(self, "SHAP plot", str(exc))

    def _on_shap_error(self, msg: str):
        self._shap_progress.setRange(0, 1)
        self._btn_shap.setEnabled(True)
        QMessageBox.critical(self, "Erreur SHAP", msg)
