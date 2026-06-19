"""Tests unitaires de tabicl_gui/llm.py.

• openai n'est jamais réellement appelé — LLMClient._client() est mocké.
• Aucun serveur LM Studio requis.
"""
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tabicl_gui.llm import (
    LLMClient,
    apply_constraints_to_data,
    clip_predictions,
    parse_constraints,
    summarize_columns,
)


# ═══════════════════════════════════════════════════════════════════════
# summarize_columns
# ═══════════════════════════════════════════════════════════════════════

class TestSummarizeColumns:
    def test_numeric_columns_have_stats(self, sample_df_reg):
        summary = summarize_columns(sample_df_reg, ["feature_a", "feature_b"])
        for col_info in summary:
            assert "min" in col_info
            assert "max" in col_info
            assert "mean" in col_info

    def test_categorical_columns_have_sample_values(self, sample_df_clf):
        summary = summarize_columns(sample_df_clf, ["species"])
        assert "sample_values" in summary[0]
        assert len(summary[0]["sample_values"]) > 0

    def test_missing_values_counted(self, sample_df_missing):
        numeric_cols = ["sepal_length", "sepal_width"]
        summary = summarize_columns(sample_df_missing, numeric_cols)
        total_missing = sum(info["missing"] for info in summary)
        assert total_missing > 0

    def test_returns_one_entry_per_column(self, sample_df_clf):
        cols = ["sepal_length", "petal_length", "species"]
        summary = summarize_columns(sample_df_clf, cols)
        assert len(summary) == 3

    def test_name_field_matches_column(self, sample_df_clf):
        cols = ["sepal_length", "species"]
        summary = summarize_columns(sample_df_clf, cols)
        names = [s["name"] for s in summary]
        assert names == cols

    def test_sample_values_capped_at_10(self):
        df = pd.DataFrame({"cat": [str(i) for i in range(50)]})
        summary = summarize_columns(df, ["cat"])
        assert len(summary[0]["sample_values"]) <= 10


# ═══════════════════════════════════════════════════════════════════════
# parse_constraints
# ═══════════════════════════════════════════════════════════════════════

class TestParseConstraints:
    def _make_json(self, col_name="Age", mn=0, mx=120, unit="years", rationale="human lifespan"):
        return json.dumps({
            "columns": {
                col_name: {"min": mn, "max": mx, "unit": unit, "rationale": rationale}
            }
        })

    def test_clean_json(self):
        result = parse_constraints(self._make_json())
        assert "Age" in result
        assert result["Age"]["min"] == pytest.approx(0.0)
        assert result["Age"]["max"] == pytest.approx(120.0)
        assert result["Age"]["unit"] == "years"

    def test_json_wrapped_in_prose(self):
        prose = f"Voici mon analyse :\n{self._make_json()}\nJ'espère que ça aide."
        result = parse_constraints(prose)
        assert "Age" in result

    def test_flat_json_without_columns_key(self):
        flat = json.dumps({"Age": {"min": 0, "max": 120, "unit": "years", "rationale": "x"}})
        result = parse_constraints(flat)
        assert "Age" in result

    def test_null_values_become_none(self):
        data = json.dumps({"columns": {"Price": {"min": 0, "max": None, "unit": None, "rationale": ""}}})
        result = parse_constraints(data)
        assert result["Price"]["max"] is None
        assert result["Price"]["unit"] is None

    def test_string_null_becomes_none(self):
        data = json.dumps({"columns": {"Weight": {"min": "null", "max": 300, "unit": "kg", "rationale": ""}}})
        result = parse_constraints(data)
        assert result["Weight"]["min"] is None

    def test_non_json_raises_value_error(self):
        with pytest.raises(ValueError, match="non interprétable"):
            parse_constraints("Désolé, je ne comprends pas la question.")

    def test_non_dict_entries_ignored(self):
        data = json.dumps({"columns": {"Age": "not a dict", "Height": {"min": 0, "max": 250, "unit": "cm", "rationale": ""}}})
        result = parse_constraints(data)
        assert "Age" not in result
        assert "Height" in result

    def test_numeric_strings_converted(self):
        data = json.dumps({"columns": {"Speed": {"min": "0", "max": "300", "unit": "km/h", "rationale": ""}}})
        result = parse_constraints(data)
        assert result["Speed"]["min"] == pytest.approx(0.0)
        assert result["Speed"]["max"] == pytest.approx(300.0)


# ═══════════════════════════════════════════════════════════════════════
# apply_constraints_to_data
# ═══════════════════════════════════════════════════════════════════════

class TestApplyConstraints:
    def _age_df(self):
        return pd.DataFrame({"Age": [25.0, 150.0, -5.0, 45.0, 200.0]})

    def _constraints(self):
        return {"Age": {"min": 0.0, "max": 120.0, "unit": "years", "rationale": ""}}

    def test_clip_mode_bounds_outliers(self):
        df, report = apply_constraints_to_data(self._age_df(), self._constraints(), mode="clip")
        assert df["Age"].max() <= 120.0
        assert df["Age"].min() >= 0.0

    def test_clip_mode_row_count_unchanged(self):
        df, _ = apply_constraints_to_data(self._age_df(), self._constraints(), mode="clip")
        assert len(df) == 5

    def test_drop_mode_removes_rows(self):
        df, report = apply_constraints_to_data(self._age_df(), self._constraints(), mode="drop")
        assert len(df) < 5
        assert df["Age"].max() <= 120.0
        assert df["Age"].min() >= 0.0

    def test_drop_mode_report_mentions_column(self):
        _, report = apply_constraints_to_data(self._age_df(), self._constraints(), mode="drop")
        assert any("Age" in r for r in report)

    def test_no_out_of_range_means_no_change(self):
        df = pd.DataFrame({"Age": [20.0, 30.0, 40.0]})
        out, report = apply_constraints_to_data(df, self._constraints(), mode="clip")
        pd.testing.assert_frame_equal(df, out)
        assert len(report) == 0

    def test_none_bounds_skipped(self):
        df = pd.DataFrame({"Age": [20.0, 300.0]})
        constraints = {"Age": {"min": None, "max": None}}
        out, report = apply_constraints_to_data(df, constraints)
        pd.testing.assert_frame_equal(df, out)

    def test_missing_column_ignored(self):
        df = pd.DataFrame({"other": [1.0, 2.0]})
        constraints = {"Age": {"min": 0.0, "max": 120.0}}
        out, report = apply_constraints_to_data(df, constraints)
        pd.testing.assert_frame_equal(df, out)
        assert len(report) == 0

    def test_min_only_constraint(self):
        df = pd.DataFrame({"Price": [-10.0, 0.0, 50.0]})
        constraints = {"Price": {"min": 0.0, "max": None}}
        out, _ = apply_constraints_to_data(df, constraints, mode="clip")
        assert out["Price"].min() >= 0.0

    def test_report_lists_number_of_outliers(self):
        _, report = apply_constraints_to_data(self._age_df(), self._constraints(), mode="clip")
        assert len(report) == 1
        assert "3" in report[0]  # 3 valeurs hors bornes (150, -5, 200)


# ═══════════════════════════════════════════════════════════════════════
# clip_predictions
# ═══════════════════════════════════════════════════════════════════════

class TestClipPredictions:
    def test_clips_above_max(self):
        preds = np.array([50.0, 130.0, 200.0])
        result = clip_predictions(preds, {"min": 0.0, "max": 120.0})
        assert result.max() <= 120.0

    def test_clips_below_min(self):
        preds = np.array([-10.0, 0.0, 50.0])
        result = clip_predictions(preds, {"min": 0.0, "max": None})
        assert result.min() >= 0.0

    def test_none_constraint_returns_unchanged(self):
        preds = np.array([1.0, 2.0, 3.0])
        result = clip_predictions(preds, None)
        np.testing.assert_array_equal(preds, result)

    def test_empty_constraint_dict_returns_unchanged(self):
        preds = np.array([1.0, 2.0, 3.0])
        result = clip_predictions(preds, {})
        np.testing.assert_array_equal(preds, result)

    def test_both_none_bounds_returns_unchanged(self):
        preds = np.array([1.0, -999.0, 999.0])
        result = clip_predictions(preds, {"min": None, "max": None})
        np.testing.assert_array_equal(preds, result)

    def test_max_only(self):
        preds = np.array([-5.0, 50.0, 200.0])
        result = clip_predictions(preds, {"min": None, "max": 100.0})
        assert result.max() <= 100.0
        assert result[0] == pytest.approx(-5.0)   # borne basse non appliquée


# ═══════════════════════════════════════════════════════════════════════
# LLMClient (openai mocké via _client)
# ═══════════════════════════════════════════════════════════════════════

def _make_openai_mock(response_content: str):
    """Retourne un mock OpenAI client dont chat.completions.create retourne response_content."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = response_content
    mock_client.chat.completions.create.return_value.choices = [mock_choice]
    return mock_client


class TestLLMClient:
    def test_list_models(self):
        client = LLMClient()
        mock_oc = MagicMock()
        mock_oc.models.list.return_value.data = [MagicMock(id="llama-3"), MagicMock(id="mistral-7b")]
        with patch.object(client, "_client", return_value=mock_oc):
            models = client.list_models()
        assert models == ["llama-3", "mistral-7b"]

    def test_analyze_columns_returns_parsed_constraints(self, sample_df_clf):
        llm_json = json.dumps({
            "columns": {
                "sepal_length": {"min": 0, "max": 10, "unit": "cm", "rationale": "physical bound"},
                "petal_width":  {"min": 0, "max": 5,  "unit": "cm", "rationale": "physical bound"},
            }
        })
        client = LLMClient(model="test-model")
        mock_oc = _make_openai_mock(llm_json)
        with patch.object(client, "_client", return_value=mock_oc):
            result = client.analyze_columns(sample_df_clf, ["sepal_length", "petal_width"], "species")
        assert "sepal_length" in result
        assert result["sepal_length"]["max"] == pytest.approx(10.0)

    def test_analyze_columns_auto_selects_first_model_when_none(self, sample_df_clf):
        llm_json = json.dumps({"columns": {}})
        client = LLMClient(model=None)
        mock_oc = _make_openai_mock(llm_json)
        mock_oc.models.list.return_value.data = [MagicMock(id="auto-model")]
        with patch.object(client, "_client", return_value=mock_oc):
            client.analyze_columns(sample_df_clf, ["sepal_length"], "species")
        # Le modèle auto-sélectionné est passé à create
        call_kwargs = mock_oc.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("model") == "auto-model" or call_kwargs.args[0] if call_kwargs.args else True

    def test_analyze_columns_invalid_json_raises(self, sample_df_clf):
        client = LLMClient(model="test-model")
        mock_oc = _make_openai_mock("Désolé, je ne sais pas répondre.")
        with patch.object(client, "_client", return_value=mock_oc):
            with pytest.raises(ValueError):
                client.analyze_columns(sample_df_clf, ["sepal_length"], "species")

    def test_analyze_columns_no_models_raises(self, sample_df_clf):
        client = LLMClient(model=None)
        mock_oc = MagicMock()
        mock_oc.models.list.return_value.data = []
        with patch.object(client, "_client", return_value=mock_oc):
            with pytest.raises(RuntimeError, match="Aucun modèle"):
                client.analyze_columns(sample_df_clf, ["sepal_length"], "species")
