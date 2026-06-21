"""Unit tests for agents7.cli (the `agents7` console script)."""

import json

import pytest

from agents7.agents.labeler import Labeler
from agents7.cli import main
from agents7.testing import create_mock_engine

from helpers import DummyOutput


def _mock_labeler(response: str = '{"value": "ok"}') -> Labeler:
    """Build a Labeler whose engine returns a fixed response."""
    return Labeler(
        name="cli-test",
        version=1,
        engine=create_mock_engine(response),
        system_prompt="Do the thing.",
        output_model=DummyOutput,
    )


@pytest.fixture
def patched_agent(monkeypatch):
    """Patch `build_agent` to return a mock-engined Labeler."""
    agent = _mock_labeler()
    monkeypatch.setattr(
        "agents7.cli.build_agent", lambda config, agent_type: agent
    )
    return agent


# ── run ────────────────────────────────────────────────────────


def test_run_prints_model_dump(patched_agent, capsys):
    """`run` prints the structured model as JSON."""
    rc = main(["run", "-c", "cfg.yml", "-i", '{"text": "hi"}'])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {"value": "ok"}


def test_run_full_includes_response_and_metrics(patched_agent, capsys):
    """`run --full` includes text, thinking, and metrics."""
    rc = main(["run", "-c", "cfg.yml", "-i", '{"text": "hi"}', "--full"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model"] == {"value": "ok"}
    assert "response" in out
    assert "metrics" in out


def test_run_from_input_file(patched_agent, capsys, tmp_path):
    """`run --input-file` reads the input from a JSON file."""
    f = tmp_path / "in.json"
    f.write_text('{"text": "hi"}', encoding="utf-8")
    rc = main(["run", "-c", "cfg.yml", "-f", str(f)])
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == {"value": "ok"}


def test_run_rejects_non_object_input(patched_agent, capsys):
    """A bare JSON string is not a valid agent input."""
    rc = main(["run", "-c", "cfg.yml", "-i", '"just a string"'])
    assert rc == 1
    assert "JSON object" in capsys.readouterr().err


def test_run_rejects_invalid_json(patched_agent, capsys):
    """Malformed JSON input exits non-zero with a message."""
    rc = main(["run", "-c", "cfg.yml", "-i", "{not json}"])
    assert rc == 1
    assert "not valid JSON" in capsys.readouterr().err


def test_run_requires_input(capsys):
    """`run` without -i/-f is an argparse error."""
    with pytest.raises(SystemExit):
        main(["run", "-c", "cfg.yml"])


# ── eval ───────────────────────────────────────────────────────


@pytest.fixture
def golden_set(tmp_path):
    """A two-row nested golden set the mock agent passes."""
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '{"input": {"text": "a"}, "value": "ok"}\n'
        '{"input": {"text": "b"}, "value": "ok"}\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def patched_eval(monkeypatch, patched_agent):
    """Patch the agent and config reader for eval tests."""
    monkeypatch.setattr(
        "agents7.cli.load_config_dict",
        lambda config: {"output_model": "helpers.DummyOutput"},
    )
    return patched_agent


def test_eval_prints_summary(patched_eval, golden_set, capsys):
    """`eval` prints accuracy and per-field breakdown."""
    rc = main(["eval", "-c", "cfg.yml", "-d", str(golden_set)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "100.0% (2/2)" in out
    assert "value" in out


def test_eval_json_output(patched_eval, golden_set, capsys):
    """`eval --json` emits the raw metrics dict."""
    rc = main(["eval", "-c", "cfg.yml", "-d", str(golden_set), "--json"])
    assert rc == 0
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["total"] == 2
    assert metrics["passed"] == 2
    assert metrics["accuracy"] == 1.0


def test_eval_writes_report(patched_eval, golden_set, capsys, tmp_path):
    """`eval --report` writes a Markdown evaluation report."""
    report = tmp_path / "report.md"
    rc = main(
        ["eval", "-c", "cfg.yml", "-d", str(golden_set), "-r", str(report)]
    )
    assert rc == 0
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "# Evaluation Report" in text
    assert "cli-test" in text


def test_eval_explicit_fields_override(patched_eval, golden_set, capsys):
    """`--fields` overrides the derived expected fields."""
    rc = main(
        ["eval", "-c", "cfg.yml", "-d", str(golden_set), "--fields", "value"]
    )
    assert rc == 0
    assert "Fields:   value" in capsys.readouterr().out


def test_eval_empty_golden_set(patched_eval, tmp_path, capsys):
    """An empty golden set exits non-zero with a message."""
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    rc = main(["eval", "-c", "cfg.yml", "-d", str(path)])
    assert rc == 1
    assert "empty" in capsys.readouterr().err


# ── parser ─────────────────────────────────────────────────────


def test_no_command_errors(capsys):
    """Invoking with no subcommand is an argparse error."""
    with pytest.raises(SystemExit):
        main([])
