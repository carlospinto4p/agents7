"""Structured call logging for agent operations.

Provides ``CallRecord`` (a frozen dataclass capturing
per-call metadata) and ``JsonCallLogger`` (a built-in
logger that emits JSON lines to a file or stream).

To enable call logging, pass a ``call_logger`` callable
to any ``Labeler``-based agent::

    from agents7.call_log import JsonCallLogger

    logger = JsonCallLogger("calls.jsonl")
    agent = Labeler(..., call_logger=logger)

Each ``act()`` / ``act_async()`` call emits one
``CallRecord`` to the logger.
"""

import json
import sys
import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO


@dataclass(frozen=True)
class CallRecord:
    """Structured record of a single agent call.

    Attributes:
        trace_id (str): Unique identifier for this call.
        agent_name (str): Name of the agent.
        agent_version (int): Version of the agent.
        model_id (str): LLM model identifier
            (e.g. ``"ollama::qwen3"``).
        success (bool): Whether parsing produced a model.
        tokens (int): Completion tokens in the response.
        prompt_tokens (int): Prompt tokens sent.
        total_duration (float): Total duration in seconds.
        error (str | None): Error message, or None.
    """

    trace_id: str
    agent_name: str
    agent_version: int
    model_id: str
    success: bool
    tokens: int = 0
    prompt_tokens: int = 0
    total_duration: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        """Convert to a plain dictionary."""
        return asdict(self)


def generate_trace_id() -> str:
    """Generate a short unique trace ID."""
    return uuid.uuid4().hex[:12]


class JsonCallLogger:
    """Logger that writes ``CallRecord`` as JSON lines.

    Writes to a file path or any writable stream
    (e.g. ``sys.stdout``).

    Usage::

        # Log to file
        logger = JsonCallLogger("calls.jsonl")

        # Log to stdout
        logger = JsonCallLogger(stream=sys.stdout)

        agent = Labeler(..., call_logger=logger)

    Attributes:
        path (str | Path | None): Output file path. Each
            call appends one JSON line.
        stream (IO | None): Writable stream. Used when
            ``path`` is None.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        stream: IO | None = None,
    ):
        if path is None and stream is None:
            stream = sys.stdout
        self._path = Path(path) if path is not None else None
        self._stream = stream
        self._fh: IO | None = None

    def _writer(self) -> IO:
        """Return the writable target, opening the file lazily.

        For a file path, a single append handle is opened on
        first write and reused across calls (avoiding an
        open/close per record). The handle is created lazily so
        a logger that never writes leaves no empty file.
        """
        if self._path is not None:
            if self._fh is None:
                self._fh = self._path.open("a", encoding="utf-8")
            return self._fh
        return self._stream

    def __call__(self, record: CallRecord) -> None:
        """Write a call record as a JSON line."""
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        writer = self._writer()
        writer.write(line + "\n")
        writer.flush()

    def close(self) -> None:
        """Close the persistent file handle, if one is open.

        Safe to call multiple times. Does not close an
        externally-supplied ``stream``.
        """
        if self._fh is not None and not self._fh.closed:
            self._fh.close()
        self._fh = None

    def __enter__(self) -> "JsonCallLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class CallStats:
    """Aggregate statistics over a set of call records.

    Attributes:
        calls (int): Number of records.
        successes (int): Records with ``success=True``.
        failures (int): Records with ``success=False``.
        tokens (int): Sum of completion tokens.
        prompt_tokens (int): Sum of prompt tokens.
        total_duration (float): Sum of durations in seconds.
    """

    calls: int = 0
    successes: int = 0
    failures: int = 0
    tokens: int = 0
    prompt_tokens: int = 0
    total_duration: float = 0.0

    @property
    def success_rate(self) -> float:
        """Fraction of calls that succeeded (0.0 if none)."""
        return self.successes / self.calls if self.calls else 0.0

    @property
    def avg_duration(self) -> float:
        """Mean duration per call in seconds (0.0 if none)."""
        return self.total_duration / self.calls if self.calls else 0.0


@dataclass(frozen=True)
class CallLogSummary:
    """Summary of a call log: overall stats plus breakdowns.

    Attributes:
        overall (CallStats): Stats across every record.
        by_agent (dict[str, CallStats]): Stats keyed by
            ``agent_name``.
        by_model (dict[str, CallStats]): Stats keyed by
            ``model_id``.
        errors (dict[str, int]): Error message to occurrence
            count, for failed calls that carried an error.
    """

    overall: CallStats
    by_agent: dict[str, CallStats] = field(default_factory=dict)
    by_model: dict[str, CallStats] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)

    def format(self) -> str:
        """Render the summary as a human-readable text block.

        Note: cost is not part of ``CallRecord`` and therefore
        not aggregated here.
        """
        o = self.overall
        lines = [
            f"Call log summary: {o.calls} calls, "
            f"{o.successes} ok ({o.success_rate:.1%}), "
            f"{o.failures} failed",
            f"Tokens: {o.tokens} completion + {o.prompt_tokens} "
            f"prompt; total duration {o.total_duration:.2f}s "
            f"(avg {o.avg_duration:.2f}s)",
        ]
        if self.by_agent:
            lines.append("")
            lines.append("By agent:")
            lines.extend(self._rows(self.by_agent))
        if self.by_model:
            lines.append("")
            lines.append("By model:")
            lines.extend(self._rows(self.by_model))
        if self.errors:
            lines.append("")
            lines.append("Errors:")
            for msg, count in sorted(
                self.errors.items(), key=lambda kv: -kv[1]
            ):
                lines.append(f"  {count:>4}  {msg}")
        return "\n".join(lines)

    @staticmethod
    def _rows(stats: dict[str, "CallStats"]) -> list[str]:
        """Format breakdown rows sorted by call count desc."""
        width = max((len(k) for k in stats), default=0)
        rows = []
        for key, s in sorted(stats.items(), key=lambda kv: -kv[1].calls):
            rows.append(
                f"  {key:<{width}}  {s.calls:>4} calls  "
                f"{s.success_rate:>6.1%} ok  {s.tokens:>7} tok"
            )
        return rows


def _normalize_record(item: object) -> dict:
    """Coerce a log item to a plain dict.

    Accepts a ``CallRecord``, a dict, or a JSON string (one
    JSONL line).
    """
    if isinstance(item, CallRecord):
        return item.to_dict()
    if isinstance(item, dict):
        return item
    if isinstance(item, str):
        return json.loads(item)
    raise TypeError(
        f"Cannot interpret call-log item of type {type(item).__name__}"
    )


def _accumulate(acc: dict, rec: dict) -> None:
    """Fold one record into a mutable accumulator dict."""
    acc["calls"] += 1
    if rec.get("success"):
        acc["successes"] += 1
    else:
        acc["failures"] += 1
    acc["tokens"] += rec.get("tokens") or 0
    acc["prompt_tokens"] += rec.get("prompt_tokens") or 0
    acc["total_duration"] += rec.get("total_duration") or 0.0


def _new_acc() -> dict:
    """Return a fresh zeroed accumulator dict."""
    return {
        "calls": 0,
        "successes": 0,
        "failures": 0,
        "tokens": 0,
        "prompt_tokens": 0,
        "total_duration": 0.0,
    }


def summarize_call_log(
    source: str | Path | Iterable[object],
) -> CallLogSummary:
    """Summarize a JSONL call log into aggregate statistics.

    :param source: A path to a JSONL file written by
        ``JsonCallLogger``, or an iterable of records where
        each item is a ``CallRecord``, a dict, or a JSON
        string (one JSONL line). Blank lines are skipped.
    :return: A :class:`CallLogSummary` with overall stats and
        per-agent / per-model breakdowns plus error counts.
    :raises TypeError: If an item cannot be interpreted.

    Cost is not part of ``CallRecord`` and is therefore not
    aggregated.
    """
    if isinstance(source, (str, Path)):
        text = Path(source).read_text(encoding="utf-8")
        items: Iterable[object] = [
            ln for ln in text.splitlines() if ln.strip()
        ]
    else:
        items = source

    overall = _new_acc()
    by_agent: dict[str, dict] = defaultdict(_new_acc)
    by_model: dict[str, dict] = defaultdict(_new_acc)
    errors: dict[str, int] = defaultdict(int)

    for item in items:
        rec = _normalize_record(item)
        _accumulate(overall, rec)
        _accumulate(by_agent[rec.get("agent_name", "")], rec)
        _accumulate(by_model[rec.get("model_id", "")], rec)
        if not rec.get("success") and rec.get("error"):
            errors[rec["error"]] += 1

    return CallLogSummary(
        overall=CallStats(**overall),
        by_agent={k: CallStats(**v) for k, v in by_agent.items()},
        by_model={k: CallStats(**v) for k, v in by_model.items()},
        errors=dict(errors),
    )
