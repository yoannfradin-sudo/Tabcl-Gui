import traceback
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class CVWorker(QThread):
    """K-fold cross-validation with TabICL in a background thread.

    Emits fold-by-fold progress then the concatenated predictions
    in the original row order so ResultsTab can compute mean ± std.
    """

    progress  = pyqtSignal(str)
    fold_done = pyqtSignal(int, int)               # fold_idx, k
    finished  = pyqtSignal(object, object, object) # all_preds, all_true, last_model
    error     = pyqtSignal(str)

    def __init__(self, task, params, X, y, k=5, stratified=True, parent=None):
        super().__init__(parent)
        self.task       = task
        self.params     = params
        self.X          = X   # numpy array
        self.y          = y   # numpy array
        self.k          = k
        self.stratified = stratified
        self._stopped   = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            from sklearn.model_selection import StratifiedKFold, KFold

            if self.task == "classification":
                from tabicl import TabICLClassifier as ModelCls
            else:
                from tabicl import TabICLRegressor as ModelCls

            splitter = (
                StratifiedKFold(n_splits=self.k, shuffle=True, random_state=42)
                if (self.task == "classification" and self.stratified)
                else KFold(n_splits=self.k, shuffle=True, random_state=42)
            )

            all_preds  = np.empty(len(self.y), dtype=object)
            all_true   = np.empty(len(self.y), dtype=object)
            last_model = None

            for fold_idx, (train_idx, val_idx) in enumerate(
                splitter.split(self.X, self.y)
            ):
                if self._stopped:
                    return
                self.progress.emit(f"Fold {fold_idx + 1}/{self.k} — entraînement…")

                X_tr, X_val = self.X[train_idx], self.X[val_idx]
                y_tr, y_val = self.y[train_idx], self.y[val_idx]

                model = ModelCls(**self.params)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_val)

                all_preds[val_idx] = preds
                all_true[val_idx]  = y_val
                last_model = model

                self.fold_done.emit(fold_idx + 1, self.k)

            self.progress.emit(f"Cross-validation terminée ({self.k} folds).")
            self.finished.emit(all_preds, all_true, last_model)

        except Exception:
            self.error.emit(traceback.format_exc())

class TrainWorker(QThread):
    """Run TabICL fit + predict in a background thread."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object, object)   # model, predictions, probas
    error = pyqtSignal(str)

    def __init__(self, task, params, X_train, y_train, X_test, parent=None):
        super().__init__(parent)
        self.task = task          # "classification" or "regression"
        self.params = params      # dict of TabICL kwargs
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            self.progress.emit("Importation de TabICL…")
            if self.task == "classification":
                from tabicl import TabICLClassifier
                model = TabICLClassifier(**self.params)
            else:
                from tabicl import TabICLRegressor
                model = TabICLRegressor(**self.params)

            if self._stopped:
                return

            self.progress.emit(
                f"Entraînement sur {len(self.X_train)} échantillons "
                f"({self.X_train.shape[1]} features)…"
            )
            model.fit(self.X_train, self.y_train)

            if self._stopped:
                return

            self.progress.emit(f"Inférence sur {len(self.X_test)} échantillons…")
            preds = model.predict(self.X_test)

            probas = None
            if self.task == "classification" and hasattr(model, "predict_proba"):
                try:
                    probas = model.predict_proba(self.X_test)
                except Exception:
                    pass

            self.progress.emit("Terminé.")
            self.finished.emit(model, preds, probas)

        except Exception:
            self.error.emit(traceback.format_exc())


class LLMWorker(QThread):
    """Interroge le LLM (LM Studio) pour inférer des contraintes métier."""

    finished = pyqtSignal(object)   # constraints dict
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, client, df, columns, target, parent=None):
        super().__init__(parent)
        self.client = client
        self.df = df
        self.columns = columns
        self.target = target

    def run(self):
        try:
            self.progress.emit("Interrogation du LLM…")
            constraints = self.client.analyze_columns(self.df, self.columns, self.target)
            self.progress.emit("Analyse terminée.")
            self.finished.emit(constraints)
        except Exception:
            self.error.emit(traceback.format_exc())


class ShapWorker(QThread):
    """Compute SHAP values in background."""

    finished = pyqtSignal(object, object)   # shap_values, feature_names
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, model, X_test, feature_names, parent=None):
        super().__init__(parent)
        self.model = model
        self.X_test = X_test
        self.feature_names = feature_names

    def run(self):
        try:
            self.progress.emit("Calcul des valeurs SHAP…")
            from tabicl.explainability import get_shap_values
            shap_vals = get_shap_values(
                self.model,
                self.X_test,
                attribute_names=self.feature_names,
            )
            self.progress.emit("SHAP terminé.")
            self.finished.emit(shap_vals, self.feature_names)
        except Exception:
            self.error.emit(traceback.format_exc())


class ForecastWorker(QThread):
    """Run TabICLForecaster in background."""

    finished = pyqtSignal(object)   # pred_df
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, context_df, prediction_length, parent=None):
        super().__init__(parent)
        self.context_df = context_df
        self.prediction_length = prediction_length

    def run(self):
        try:
            self.progress.emit("Importation du forecaster…")
            from tabicl.forecast import TabICLForecaster, TimeSeriesDataFrame
            tsdf = TimeSeriesDataFrame.from_data_frame(self.context_df)
            forecaster = TabICLForecaster()
            self.progress.emit("Prévision en cours…")
            pred = forecaster.predict_df(tsdf, self.prediction_length)
            self.progress.emit("Prévision terminée.")
            self.finished.emit(pred)
        except Exception:
            self.error.emit(traceback.format_exc())


class FinetuneWorker(QThread):
    """Run FinetunedTabICL fit + predict in background."""

    progress = pyqtSignal(str)
    finished = pyqtSignal(object, object, object)   # model, preds, probas
    error = pyqtSignal(str)

    def __init__(self, task, params, X_train, y_train, X_val, y_val, X_test, parent=None):
        super().__init__(parent)
        self.task = task
        self.params = params
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test

    def run(self):
        try:
            self.progress.emit("Importation du modèle fine-tunable…")
            if self.task == "classification":
                from tabicl import FinetunedTabICLClassifier
                model = FinetunedTabICLClassifier(**self.params)
            else:
                from tabicl import FinetunedTabICLRegressor
                model = FinetunedTabICLRegressor(**self.params)

            self.progress.emit("Fine-tuning en cours…")
            model.fit(
                self.X_train, self.y_train,
                X_val=self.X_val, y_val=self.y_val,
            )

            self.progress.emit("Inférence…")
            preds = model.predict(self.X_test)

            probas = None
            if self.task == "classification" and hasattr(model, "predict_proba"):
                try:
                    probas = model.predict_proba(self.X_test)
                except Exception:
                    pass

            self.progress.emit("Fine-tuning terminé.")
            self.finished.emit(model, preds, probas)
        except Exception:
            self.error.emit(traceback.format_exc())
