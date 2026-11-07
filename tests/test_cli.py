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
    """Patch the composed route's agent and config reader.

    `eval` delegates to `run_eval_suite`, so the patches land on
    `agents7.eval.suite` rather than on the CLI module.
    """
    monkeypatch.setattr(
        "agents7.eval.suite.build_agent",
        lambda config, agent_type=None: patched_agent,
    )
    monkeypatch.setattr(
        "agents7.eval.suite.load_config_dict",
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


def test_eval_call_log(patched_eval, golden_set, capsys, tmp_path):
    """`eval --call-log` writes a log and summarises it."""
    log = tmp_path / "calls.jsonl"
    rc = main(
        [
            "eval",
            "-c",
            "cfg.yml",
            "-d",
            str(golden_set),
            "--call-log",
            str(log),
        ]
    )
    assert rc == 0
    assert log.exists()
    assert "Call log summary:" in capsys.readouterr().out


def test_eval_empty_golden_set(patched_eval, tmp_path, capsys):
    """An empty golden set exits non-zero with a message."""
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    rc = main(["eval", "-c", "cfg.yml", "-d", str(path)])
    assert rc == 1
    assert "empty" in capsys.readouterr().err


# ── chat-feedback ──────────────────────────────────────────────


@pytest.fixture
def feedback_file(tmp_path):
    """A feedback file with one shipped rule and one outstanding."""
    path = tmp_path / "feedback.yml"
    path.write_text(
        "read_id: r\n"
        "agent: demo\n"
        "agent_version: '1.0.0'\n"
        "entries:\n"
        "  - conversation: warm\n"
        "    turn: 1\n"
        "    verdict: wrong\n"
        "    he_said: mucho mejor\n"
        "    should_say: mas entretenido\n"
        "    why: overstates it\n"
        "    rule: no overstating\n"
        "    lands_in: david.yml\n"
        "  - conversation: warm\n"
        "    turn: 2\n"
        "    verdict: wrong\n"
        "    why: swearing\n"
        "    rule: no swearing\n"
        "    lands_in: david.yml\n"
        "    shipped: v0.1.5\n",
        encoding="utf-8",
    )
    return path


def test_chat_feedback_prints_the_report_by_default(feedback_file, capsys):
    assert main(["chat-feedback", str(feedback_file)]) == 0
    out = capsys.readouterr().out
    assert "# Feedback — demo v1.0.0" in out
    assert "no overstating" in out


def test_chat_feedback_writes_the_report(feedback_file, tmp_path, capsys):
    target = tmp_path / "out" / "report.md"
    main(["chat-feedback", str(feedback_file), "--report", str(target)])
    assert "no overstating" in target.read_text(encoding="utf-8")
    assert "Wrote" in capsys.readouterr().out


def test_chat_feedback_writes_eval_rows(feedback_file, tmp_path, capsys):
    target = tmp_path / "rows.jsonl"
    main(["chat-feedback", str(feedback_file), "--rows", str(target)])
    rows = [
        json.loads(line)
        for line in target.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["target"] == "mas entretenido"


def test_chat_feedback_counts_what_has_not_shipped(
    feedback_file, tmp_path, capsys
):
    main(
        [
            "chat-feedback",
            str(feedback_file),
            "--rows",
            str(tmp_path / "rows.jsonl"),
        ]
    )
    assert "1 rule(s) not shipped yet." in capsys.readouterr().out


# ── parser ─────────────────────────────────────────────────────


def test_no_command_errors(capsys):
    """Invoking with no subcommand is an argparse error."""
    with pytest.raises(SystemExit):
        main([])
