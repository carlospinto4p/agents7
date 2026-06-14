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
from dataclasses import asdict, dataclass
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
