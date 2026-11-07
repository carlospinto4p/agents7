"""Command-line interface for agents7.

Exposes the ``agents7`` console script with two subcommands:

- ``agents7 run`` — execute a YAML-configured agent against a
  single input (inline JSON or a file) and print the structured
  output.
- ``agents7 eval`` — evaluate a YAML-configured agent against a
  labeled golden set, print metrics, and optionally write a
  Markdown report and a call log.

``run`` is thin glue over the framework's YAML loaders; ``eval``
is a thin wrapper over `agents7.eval.run_eval_suite`, the same
composed route a Python caller uses. Both let you run and score
agents headlessly without spinning up the UI.
"""

import argparse
import json
import sys
from pathlib import Path

from agents7 import __version__
from agents7.eval.data import build_agent
from agents7.eval.suite import run_eval_suite
from agents7.eval.trajectory import TrajectoryMode
from agents7.exceptions import Agents7Exception


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    :return: Configured `ArgumentParser` with ``run`` and
        ``eval`` subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="agents7",
        description=("Run and evaluate YAML-configured agents7 agents."),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"agents7 {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser(
        "run",
        help="Run an agent against a single input.",
    )
    _add_common_agent_args(run_p)
    input_group = run_p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i",
        "--input",
        help="Input as a JSON object (or list of objects).",
    )
    input_group.add_argument(
        "-f",
        "--input-file",
        type=Path,
        help="Path to a JSON file holding the input.",
    )
    run_p.add_argument(
        "--full",
        action="store_true",
        help="Print the full response (text, thinking, metrics).",
    )
    run_p.set_defaults(func=_cmd_run)

    eval_p = subparsers.add_parser(
        "eval",
        help="Evaluate an agent against a golden set.",
    )
    _add_common_agent_args(eval_p)
    eval_p.add_argument(
        "-d",
        "--data",
        type=Path,
        required=True,
        help="Path to the golden set (.jsonl, .json, or .csv).",
    )
    eval_p.add_argument(
        "--fields",
        help=(
            "Comma-separated expected field names. If omitted, "
            "derived from the golden set and output model."
        ),
    )
    eval_p.add_argument(
        "-r",
        "--report",
        type=Path,
        help="Write a Markdown evaluation report to this path.",
    )
    eval_p.add_argument(
        "--confusion-field",
        help=(
            "Categorical field for the confusion matrix in the "
            "report (defaults to 'category' when present)."
        ),
    )
    eval_p.add_argument(
        "--call-log",
        type=Path,
        help=(
            "Log every call made during the run as JSON lines to "
            "this path, and summarise them in the output."
        ),
    )
    eval_p.add_argument(
        "--trajectory-mode",
        choices=[m.value for m in TrajectoryMode],
        help=(
            "For a tool agent, how strictly to match the tool "
            "calls: in_order tolerates extra calls, exact does "
            "not. Defaults to the config's trajectory_mode."
        ),
    )
    eval_p.add_argument(
        "--json",
        action="store_true",
        help="Print the metrics as JSON instead of a summary.",
    )
    eval_p.set_defaults(func=_cmd_eval)

    review_p = subparsers.add_parser(
        "chat-review",
        help=(
            "Render a chat agent's read as an HTML page from saved "
            "recordings. Makes no LLM calls."
        ),
    )
    review_p.add_argument(
        "spec",
        type=Path,
        help="The read's YAML spec: which recordings, and the notes.",
    )
    review_p.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        help="Where to write the page.",
    )
    review_p.add_argument(
        "--history",
        type=Path,
        help=(
            "Append this read's figures to a history file, so the "
            "next read's comparison column fills itself."
        ),
    )
    review_p.add_argument(
        "--read-id",
        help="Name for the history entry. Defaults to the spec's name.",
    )
    review_p.set_defaults(func=_cmd_chat_review)

    feedback_p = subparsers.add_parser(
        "chat-feedback",
        help=(
            "Turn a feedback file into the rule changes, the eval "
            "rows and the operator's report. Makes no LLM calls."
        ),
    )
    feedback_p.add_argument(
        "spec",
        type=Path,
        help="The feedback YAML: corrections and the rules they became.",
    )
    feedback_p.add_argument(
        "--report",
        type=Path,
        help="Write the Markdown report here.",
    )
    feedback_p.add_argument(
        "--rows",
        type=Path,
        help="Write the machine-readable eval rows here, as JSONL.",
    )
    feedback_p.set_defaults(func=_cmd_chat_feedback)

    return parser


def _add_common_agent_args(parser: argparse.ArgumentParser) -> None:
    """Add the ``--config`` / ``--agent-type`` args shared by
    both subcommands.

    :param parser: Subparser to extend.
    """
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        required=True,
        help="Path to the agent's YAML config file.",
    )
    parser.add_argument(
        "-t",
        "--agent-type",
        help=(
            "Agent type (labeler, detector, classifier, "
            "regressor, tool). Defaults to the config's "
            "agent_type."
        ),
    )


def _load_run_input(args: argparse.Namespace) -> dict | list:
    """Parse and validate the input for the ``run`` subcommand.

    :param args: Parsed CLI arguments.
    :return: A dict or list of dicts to pass to ``agent.act``.
    :raises ValueError: If the input is not valid JSON or not a
        dict / list of dicts.
    """
    if args.input_file is not None:
        raw = Path(args.input_file).read_text(encoding="utf-8")
    else:
        raw = args.input
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Input is not valid JSON: {exc}") from exc
    if isinstance(value, dict):
        return value
    if isinstance(value, list) and all(
        isinstance(item, dict) for item in value
    ):
        return value
    raise ValueError("Input must be a JSON object or a list of JSON objects.")


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute the ``run`` subcommand.

    :param args: Parsed CLI arguments.
    :return: Process exit code (0 on success).
    """
    input_data = _load_run_input(args)
    agent = build_agent(args.config, args.agent_type)
    response = agent.act(input_data)

    model_dump = (
        response.model.model_dump() if response.model is not None else None
    )
    if args.full:
        payload = {
            "response": response.response,
            "model": model_dump,
            "thinking": response.thinking,
            "metrics": response.metrics.model_dump(),
        }
    else:
        payload = model_dump
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    """Execute the ``eval`` subcommand.

    Thin wrapper over `run_eval_suite` — the composed route does
    the loading, scoring, metrics, call-log summary, and report.

    :param args: Parsed CLI arguments.
    :return: Process exit code (0 on success).
    """
    fields = (
        [f.strip() for f in args.fields.split(",") if f.strip()]
        if args.fields
        else None
    )
    run = run_eval_suite(
        args.config,
        args.data,
        agent_type=args.agent_type,
        fields=fields,
        report=args.report,
        confusion_field=args.confusion_field,
        call_log=args.call_log,
        trajectory_mode=args.trajectory_mode,
    )

    if args.json:
        print(json.dumps(run.metrics, indent=2, default=str))
    else:
        print(run.format())
    return 0


def _cmd_chat_review(args: argparse.Namespace) -> int:
    """Render a read's page from its spec.

    :param args: Parsed arguments.
    :return: Process exit code.
    """
    from agents7.eval.chat_review import write_review

    path = write_review(
        args.spec,
        args.out,
        history=args.history,
        read_id=args.read_id,
    )
    print(f"Wrote {path}")
    if args.history:
        print(f"Recorded this read's figures in {args.history}")
    return 0


def _cmd_chat_feedback(args: argparse.Namespace) -> int:
    """Derive the outputs of a feedback file.

    With neither `--report` nor `--rows` it prints the report, so
    the plain invocation still shows something useful.

    :param args: Parsed arguments.
    :return: Process exit code.
    """
    from agents7.eval.chat_feedback import (
        load_feedback,
        render_feedback,
        write_eval_rows,
        write_feedback_report,
    )

    feedback = load_feedback(args.spec)
    if not args.report and not args.rows:
        print(render_feedback(feedback), end="")
        return 0
    if args.report:
        print(f"Wrote {write_feedback_report(feedback, args.report)}")
    if args.rows:
        path = write_eval_rows(feedback, args.rows)
        print(f"Wrote {len(feedback.eval_rows())} eval rows to {path}")
    outstanding = sum(
        len(change.outstanding) for change in feedback.rule_changes()
    )
    if outstanding:
        print(f"{outstanding} rule(s) not shipped yet.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``agents7`` console script.

    :param argv: Optional argument list (defaults to
        ``sys.argv[1:]``).
    :return: Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (Agents7Exception, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
