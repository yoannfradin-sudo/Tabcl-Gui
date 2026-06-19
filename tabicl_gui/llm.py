"""Couche LLM (compatible OpenAI / LM Studio) pour injecter des
connaissances métier dans le pipeline TabICL.

Le LLM analyse la description des colonnes et propose des contraintes
métier (bornes physiques, unité, valeurs impossibles). Ces contraintes
sont ensuite appliquées pour nettoyer les données d'entraînement et/ou
borner les prédictions de régression.
"""
import json
import re
import numpy as np
import pandas as pd

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_API_KEY = "lm-studio"


# ----------------------------------------------------------------------
# Construction du prompt
# ----------------------------------------------------------------------

def summarize_columns(df: pd.DataFrame, columns) -> list:
    """Résumé compact des colonnes pour le prompt LLM."""
    summary = []
    for col in columns:
        s = df[col]
        info = {"name": col, "dtype": str(s.dtype), "missing": int(s.isnull().sum())}
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            info["min"] = float(np.nanmin(s.values))
            info["max"] = float(np.nanmax(s.values))
            info["mean"] = round(float(np.nanmean(s.values)), 4)
        else:
            vals = s.dropna().unique()[:10]
            info["sample_values"] = [str(v) for v in vals]
        summary.append(info)
    return summary


SYSTEM_PROMPT = (
    "Tu es un expert métier en qualité de données. On te fournit la description "
    "des colonnes d'un jeu de données tabulaire (nom, type, statistiques, "
    "exemples). Pour chaque colonne pertinente, déduis des contraintes métier "
    "RÉALISTES : bornes physiques plausibles (min/max), unité, et une courte "
    "justification. Par exemple une colonne 'Âge' (humain) est typiquement "
    "comprise entre 0 et 120 ans ; une distance ou un prix ne peut être négatif. "
    "Réponds UNIQUEMENT par un objet JSON valide, sans texte autour."
)


def build_user_prompt(summary: list, target: str) -> str:
    schema = {
        "columns": {
            "<nom_colonne>": {
                "min": "nombre ou null (borne basse plausible)",
                "max": "nombre ou null (borne haute plausible)",
                "unit": "chaîne ou null",
                "rationale": "courte justification métier",
            }
        }
    }
    return (
        f"Colonne cible à prédire : {target}\n\n"
        f"Colonnes :\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n\n"
        f"Renvoie un objet JSON respectant exactement ce schéma :\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        "N'inclus que les colonnes numériques pour lesquelles une borne métier "
        "est pertinente. Utilise null quand une borne n'a pas de sens."
    )


# ----------------------------------------------------------------------
# Client LLM
# ----------------------------------------------------------------------

class LLMClient:
    def __init__(self, base_url=DEFAULT_BASE_URL, api_key=DEFAULT_API_KEY, model=None):
        self.base_url = base_url
        self.api_key = api_key or DEFAULT_API_KEY
        self.model = model

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Le paquet 'openai' est requis pour la couche LLM. "
                "Installez-le : pip install openai"
            ) from exc
        return OpenAI(base_url=self.base_url, api_key=self.api_key)

    def list_models(self) -> list:
        client = self._client()
        return [m.id for m in client.models.list().data]

    def analyze_columns(self, df: pd.DataFrame, columns, target: str) -> dict:
        summary = summarize_columns(df, columns)
        client = self._client()
        model = self.model
        if not model:
            models = self.list_models()
            if not models:
                raise RuntimeError("Aucun modèle chargé dans LM Studio.")
            model = models[0]

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(summary, target)},
            ],
            temperature=0.1,
        )
        content = resp.choices[0].message.content
        return parse_constraints(content)


# ----------------------------------------------------------------------
# Parsing & application des contraintes
# ----------------------------------------------------------------------

def parse_constraints(text: str) -> dict:
    """Extrait un dict de contraintes depuis la réponse LLM (robuste)."""
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError(f"Réponse LLM non interprétable en JSON :\n{text[:500]}")
        data = json.loads(m.group(0))

    cols = data.get("columns", data)
    out = {}
    for name, c in cols.items():
        if not isinstance(c, dict):
            continue
        out[name] = {
            "min": _num(c.get("min")),
            "max": _num(c.get("max")),
            "unit": c.get("unit"),
            "rationale": c.get("rationale", ""),
        }
    return out


def _num(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def apply_constraints_to_data(df: pd.DataFrame, constraints: dict, mode: str = "clip"):
    """Applique les contraintes aux données.

    mode='clip' : borne les valeurs hors limites.
    mode='drop' : supprime les lignes hors limites.
    Retourne (df_nettoyé, rapport:list[str]).
    """
    out = df.copy()
    report = []
    keep_mask = pd.Series(True, index=out.index)

    for col, c in constraints.items():
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        lo, hi = c.get("min"), c.get("max")
        if lo is None and hi is None:
            continue
        below = (out[col] < lo) if lo is not None else pd.Series(False, index=out.index)
        above = (out[col] > hi) if hi is not None else pd.Series(False, index=out.index)
        n = int((below | above).sum())
        if n == 0:
            continue
        if mode == "drop":
            keep_mask &= ~(below | above)
            report.append(f"{col} : {n} ligne(s) hors bornes [{lo}, {hi}] supprimée(s)")
        else:
            out[col] = out[col].clip(lower=lo, upper=hi)
            report.append(f"{col} : {n} valeur(s) bornée(s) à [{lo}, {hi}]")

    if mode == "drop":
        out = out[keep_mask]
    return out, report


def clip_predictions(preds, target_constraint):
    """Borne les prédictions de régression selon la contrainte de la cible."""
    if not target_constraint:
        return preds
    lo = target_constraint.get("min")
    hi = target_constraint.get("max")
    if lo is None and hi is None:
        return preds
    return np.clip(
        np.asarray(preds, dtype=float),
        lo if lo is not None else -np.inf,
        hi if hi is not None else np.inf,
    )
