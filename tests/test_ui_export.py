"""Unit tests for agents7.ui.evaluation.export (CSV + JSONL)."""

import csv
import io
import json

from agents7.ui.evaluation.runner import EvalResult
from agents7.ui.evaluation.export import (
    format_results_csv,
    format_results_jsonl,
    _input_to_str,
)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------


def _result_passed() -> EvalResult:
    return EvalResult(
        input_data={"text": "good input"},
        expected={"category": "POSITIVE", "reason": "HAPPY"},
        actual={"category": "POSITIVE", "reason": "HAPPY"},
        passed=True,
        field_results={"category": True, "reason": True},
    )


def _result_failed() -> EvalResult:
    return EvalResult(
        input_data={"text": "bad input"},
        expected={"category": "POSITIVE", "reason": "HAPPY"},
        actual={"category": "NEGATIVE", "reason": "SAD"},
        passed=False,
        field_results={"category": False, "reason": False},
    )


def _result_none_actual() -> EvalResult:
    return EvalResult(
        input_data={"text": "broken input"},
        expected={"category": "POSITIVE", "reason": "HAPPY"},
        actual=None,
        passed=False,
        field_results={"category": False, "reason": False},
        error="Model parsing failed",
    )


def _result_error() -> EvalResult:
    return EvalResult(
        input_data={"text": "error input"},
        expected={"category": "POSITIVE", "reason": "HAPPY"},
        actual=None,
        passed=False,
        field_results={},
        error="Connection refused",
    )


FIELDS = ["category", "reason"]


# ----------------------------------------------------------------
# format_results_csv
# ----------------------------------------------------------------


class TestFormatResultsCsv:
    def test_header_columns(self):
        csv_str = format_results_csv([], FIELDS)
        reader = csv.reader(io.StringIO(csv_str))
        header = next(reader)
        assert header == [
            "input",
            "category_expected",
            "category_actual",
            "category_passed",
            "reason_expected",
            "reason_actual",
            "reason_passed",
            "passed",
            "error",
        ]

    def test_empty_results(self):
        csv_str = format_results_csv([], FIELDS)
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 1  # header only

    def test_passed_row(self):
        csv_str = format_results_csv([_result_passed()], FIELDS)
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)
        assert row["input"] == "good input"
        assert row["category_expected"] == "POSITIVE"
        assert row["category_actual"] == "POSITIVE"
        assert row["category_passed"] == "True"
        assert row["passed"] == "True"
        assert row["error"] == ""

    def test_failed_row(self):
        csv_str = format_results_csv([_result_failed()], FIELDS)
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)
        assert row["category_actual"] == "NEGATIVE"
        assert row["category_passed"] == "False"
        assert row["passed"] == "False"

    def test_none_actual_row(self):
        csv_str = format_results_csv([_result_none_actual()], FIELDS)
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)
        assert row["category_actual"] == ""
        assert row["reason_actual"] == ""
        assert row["error"] == "Model parsing failed"

    def test_error_row(self):
        csv_str = format_results_csv([_result_error()], FIELDS)
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)
        assert row["error"] == "Connection refused"
        assert row["category_passed"] == "False"

    def test_multiple_rows(self):
        results = [_result_passed(), _result_failed()]
        csv_str = format_results_csv(results, FIELDS)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["passed"] == "True"
        assert rows[1]["passed"] == "False"

    def test_single_field(self):
        result = EvalResult(
            input_data={"text": "test"},
            expected={"reason": "OK"},
            actual={"reason": "OK"},
            passed=True,
            field_results={"reason": True},
        )
        csv_str = format_results_csv([result], ["reason"])
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)
        assert "reason_expected" in row
        assert "category_expected" not in row

    def test_multi_key_input_dict(self):
        result = EvalResult(
            input_data={"text": "hello", "id": 42},
            expected={"category": "X"},
            actual={"category": "X"},
            passed=True,
            field_results={"category": True},
        )
        csv_str = format_results_csv([result], ["category"])
        reader = csv.DictReader(io.StringIO(csv_str))
        row = next(reader)
        assert "hello" in row["input"]
        assert "42" in row["input"]


# ----------------------------------------------------------------
# format_results_jsonl
# ----------------------------------------------------------------


class TestFormatResultsJsonl:
    def test_empty_results(self):
        assert format_results_jsonl([]) == ""

    def test_one_line_per_result(self):
        results = [_result_passed(), _result_failed()]
        jsonl_str = format_results_jsonl(results)
        lines = jsonl_str.strip().split("\n")
        assert len(lines) == 2

    def test_passed_row_keys(self):
        jsonl_str = format_results_jsonl([_result_passed()])
        obj = json.loads(jsonl_str)
        assert set(obj.keys()) == {
            "input",
            "expected",
            "actual",
            "field_results",
            "passed",
            "error",
        }

    def test_passed_row_values(self):
        jsonl_str = format_results_jsonl([_result_passed()])
        obj = json.loads(jsonl_str)
        assert obj["passed"] is True
        assert obj["error"] is None
        assert obj["actual"]["category"] == "POSITIVE"

    def test_none_actual(self):
        jsonl_str = format_results_jsonl([_result_none_actual()])
        obj = json.loads(jsonl_str)
        assert obj["actual"] is None
        assert obj["error"] == "Model parsing failed"

    def test_error_row(self):
        jsonl_str = format_results_jsonl([_result_error()])
        obj = json.loads(jsonl_str)
        assert obj["error"] == "Connection refused"
        assert obj["passed"] is False

    def test_excludes_response_key(self):
        jsonl_str = format_results_jsonl([_result_passed()])
        obj = json.loads(jsonl_str)
        assert "response" not in obj

    def test_valid_json_per_line(self):
        results = [
            _result_passed(),
            _result_failed(),
            _result_error(),
        ]
        jsonl_str = format_results_jsonl(results)
        for line in jsonl_str.strip().split("\n"):
            json.loads(line)  # should not raise


# ----------------------------------------------------------------
# _input_to_str
# ----------------------------------------------------------------


class TestInputToStr:
    def test_single_text_key(self):
        assert _input_to_str({"text": "hello"}) == "hello"

    def test_multi_key_dict(self):
        result = _input_to_str({"text": "hi", "id": 1})
        assert "hi" in result
        assert "1" in result

    def test_non_dict(self):
        assert _input_to_str("plain string") == "plain string"
