import io
import pandas as pd
import numpy as np
import requests
from pathlib import Path


def load_data(path_or_url: str) -> pd.DataFrame:
    """Load tabular data from a local file path or a remote URL."""
    src = path_or_url.strip()
    if src.startswith("http://") or src.startswith("https://"):
        resp = requests.get(src, timeout=30)
        resp.raise_for_status()
        content = resp.content
        name = src.split("?")[0].split("/")[-1].lower()
        if name.endswith(".xlsx") or name.endswith(".xls"):
            return pd.read_excel(io.BytesIO(content))
        if name.endswith(".json"):
            return pd.read_json(io.BytesIO(content))
        if name.endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(content))
        return pd.read_csv(io.BytesIO(content))

    p = Path(src)
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(src)
    if ext == ".json":
        return pd.read_json(src)
    if ext == ".parquet":
        return pd.read_parquet(src)
    return pd.read_csv(src)


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score, classification_report,
        confusion_matrix,
    )

    classes = np.unique(y_true)
    n_classes = len(classes)
    avg = "binary" if n_classes == 2 else "macro"

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
    }

    if y_proba is not None:
        try:
            if n_classes == 2:
                proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                metrics["roc_auc"] = roc_auc_score(y_true, proba)
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average="macro"
                )
        except Exception:
            pass

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
    metrics["classes"] = classes
    metrics["report"] = classification_report(y_true, y_pred, zero_division=0)
    return metrics


def compute_regression_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mse),
    }


def detect_task_type(series: pd.Series) -> str:
    """Heuristic: if target is numeric with many unique values → regression."""
    if pd.api.types.is_bool_dtype(series):
        return "classification"
    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        if n_unique <= 20 or (n_unique / len(series) < 0.05):
            return "classification"
        return "regression"
    return "classification"
