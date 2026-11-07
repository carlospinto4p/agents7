"""Structured call logging for agent operations.

Provides ``CallRecord`` (a frozen dataclass capturing
per-call metadata) and ``JsonCallLogger`` (a built-in
logger that emits JSON lines to a file or stream).

To enable call logging, pass a ``call_logger`` callable
to any ``Labeler``-based agent, to a ``Chatter``, or to a
``ToolAgent``::

    from agents7.call_log import JsonCallLogger

    logger = JsonCallLogger("calls.jsonl")
    agent = Labeler(..., call_logger=logger)

One ``CallRecord`` is emitted per LLM call. An ``act()``,
``act_async()``, ``act_stream()`` or ``act_stream_async()``
that parsed first time made one call and so emits one record;
an ``act()`` or ``act_async()`` that needed parse retries
emits one per attempt, sharing the answer's ``trace_id``. A
streamed call emits its record once the caller has finished
iterating, since the outcome is not known before then.

A chat turn emits one record per LLM call it makes, which is
more than one whenever the turn is guarded: the reply, the
guard's check, and one more pair for every regeneration the
guard asked for. They share the turn's ``trace_id``.

Above the turn sits ``run_id``, shared by every call of one
conversation or one eval run. A ``Chatter`` fills it from its
``conversation_id``; every other agent takes a ``run_id`` from
the caller, and leaves it None when none was given.
"""

import json
import sqlite3
import sys
import threading
import uuid
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

from agents7.exceptions import AgentConfigurationError


def now() -> str:
    """Return the current UTC time as an ISO-8601 string.

    What every `CallRecord` producer stamps its ``recorded_at``
    with. Exposed so the producers share one definition of the
    format rather than each choosing its own.
    """
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class CallRecord:
    """Structured record of a single agent call.

    Attributes:
        trace_id (str): Identifier for the turn this call
            belongs to. Every call of one turn carries the same
            id, so a consumer can group them and read what the
            whole turn cost. A structured-output agent that
            parsed first time made one call, so its id is
            unique to that record; one that needed parse
            retries shares it across the attempts. A guarded
            chat turn makes at least two — the reply and the
            guard's check.
        recorded_at (str | None): ISO-8601 UTC timestamp, stamped
            by the producer when the record is built — which is
            immediately after the call returned. A log without
            one can be read in order and cannot be read against a
            date, so it cannot answer what a week cost or whether
            a model got slower, which is what
            `measurement-needs-history` asks of every figure this
            framework produces. Every producer calls `now()` for
            it rather than a default doing so, so that a row read
            back from a log written before this field existed
            reads as unstamped instead of as having happened at
            the moment it was read.
        run_id (str | None): Identifier for the whole run the
            turn belongs to — a conversation, an eval run, a
            benchmark sweep. `trace_id` groups the calls of one
            turn and nothing grouped the turns, so a log could
            say what a turn cost and not what a conversation
            cost. None means nobody named a run, which is not
            the same as every call belonging to one.
        agent_name (str): Name of the agent.
        agent_version (int): Version of the agent.
        model_id (str): The model the agent was *configured*
            with (e.g. ``"ollama::qwen3"``).
        model (str | None): The model that actually *answered*,
            as the provider named it in its reply. Usually the
            same as ``model_id`` and deliberately kept separate,
            because the two diverge exactly where it matters: an
            endpoint resolving an alias (`deepseek-chat` is
            served by `deepseek-v4-flash`), and a
            ``fallback_model`` answering after the primary
            failed. Every record used to carry only the
            configured name, so a run half-answered by a fallback
            logged all of it against the model that failed. None
            when the provider named nothing back.
        success (bool): Whether the call produced a usable
            result — a parsed model for a structured-output
            agent, a returned reply for a chat one.
        tokens (int): Completion tokens in the response.
        prompt_tokens (int): Prompt tokens sent.
        cached_prompt_tokens (int): Prompt tokens the provider
            served from its own cache.
        total_duration (float): Total duration in seconds.
        cost_cents (float | None): Estimated cost, or None when
            the engine declares no pricing.
        cached (bool): Whether the response came from the
            response cache instead of the provider. A cached
            call carries the token counts of the call that
            filled the cache but was not billed for them, so a
            consumer summing tokens needs this to interpret the
            total.
        retries (int): How many attempts at this call failed and
            were retried. A run that hit rate limits and retried
            its way through used to look in the log exactly like
            one that succeeded first time, while having taken
            several times as long.
        retry_duration (float): Seconds those failed attempts and
            their backoff waits took. Separate from
            ``total_duration``, which is the successful attempt's
            own figure.
        error (str | None): Error message, or None.
        prompt (str | None): The messages that were sent, as a
            JSON string, on a call that failed. None on a call
            that succeeded.

    Every field `LLMMetrics` carries is mirrored here. A record
    that drops one silently makes that measurement unavailable to
    every consumer of the log — which is what happened to
    `cost_cents`, so an eval run could not report what it spent.

    `prompt` is the one field deliberately not carried by every
    record. A successful call's prompt is derivable from the
    agent's config; a failed one's is the evidence needed to
    reproduce it by hand, and it cannot be recovered afterwards
    because a re-run at temperature produces a different call.
    Recording it on every row would also multiply the log by the
    size of the system prompt — for a chat agent resending a
    whole persona each turn, by around a hundred to one.
    """

    trace_id: str
    agent_name: str
    agent_version: int
    model_id: str
    success: bool
    tokens: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    total_duration: float = 0.0
    cost_cents: float | None = None
    cached: bool = False
    model: str | None = None
    retries: int = 0
    retry_duration: float = 0.0
    error: str | None = None
    prompt: str | None = None
    run_id: str | None = None
    recorded_at: str | None = None

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


_BOOL_COLUMNS = ("success", "cached")

# Column order follows `CallRecord`'s fields, so the two cannot
# drift silently: `_column_names` reads them off the dataclass and
# a field added without a column here fails the schema check on
# the next write.
_CALLS_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    row_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id             TEXT    NOT NULL,
    agent_name           TEXT    NOT NULL,
    agent_version        INTEGER NOT NULL,
    model_id             TEXT    NOT NULL,
    success              INTEGER NOT NULL,
    tokens               INTEGER NOT NULL,
    prompt_tokens        INTEGER NOT NULL,
    cached_prompt_tokens INTEGER NOT NULL,
    total_duration       REAL    NOT NULL,
    cost_cents           REAL,
    cached               INTEGER NOT NULL,
    model                TEXT,
    retries              INTEGER NOT NULL,
    retry_duration       REAL    NOT NULL,
    error                TEXT,
    prompt               TEXT,
    run_id               TEXT,
    recorded_at          TEXT
);
CREATE INDEX IF NOT EXISTS calls_run_id ON calls(run_id);
CREATE INDEX IF NOT EXISTS calls_trace_id ON calls(trace_id);
CREATE INDEX IF NOT EXISTS calls_recorded_at ON calls(recorded_at);
"""


def _column_names() -> tuple[str, ...]:
    """Return the record's field names, in declaration order."""
    return tuple(f.name for f in fields(CallRecord))


def _declared_columns() -> dict[str, str]:
    """Read the column names and types out of `_CALLS_SCHEMA`.

    Taken from the DDL rather than listed a second time, so the
    migration below cannot fall behind the table it migrates.

    :return: Column name to its declared type.
    """
    body = _CALLS_SCHEMA.split("(", 1)[1].split(");", 1)[0]
    columns: dict[str, str] = {}
    for line in body.splitlines():
        parts = line.strip().rstrip(",").split()
        if len(parts) >= 2 and parts[0] != "row_id":
            columns[parts[0]] = parts[1]
    return columns


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    """Add columns an older database predates.

    ``CREATE TABLE IF NOT EXISTS`` leaves an existing table
    exactly as it was, so a version that adds a field writes an
    INSERT naming a column the file has never heard of and every
    write fails. A log written by an older version has to keep
    accepting rows — `model` was the first field to hit this, and
    the point of doing it generally is that it will not be the
    last.

    New columns are added nullable, which is both what SQLite
    allows without a default and the honest value: the rows
    already in the file were written before anything measured
    this, and ``NULL`` says so where a zero would not.

    :param conn: An open connection to the log database.
    """
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(calls)").fetchall()
    }
    for name, sql_type in _declared_columns().items():
        if name not in existing:
            conn.execute(f"ALTER TABLE calls ADD COLUMN {name} {sql_type}")


class SqliteCallLogger:
    """Logger that writes ``CallRecord`` rows to SQLite.

    The durable counterpart to `JsonCallLogger`. A JSON-lines file
    suits one eval run: reading it means parsing the whole file,
    and two processes appending at once interleave their lines.
    A chat agent logging every turn for months wants neither.

    One row per LLM call, in a ``calls`` table whose columns are
    `CallRecord`'s fields — so a query can ask what a run cost, or
    what a model averaged last week, without loading the log into
    memory. Indexed on ``run_id``, ``trace_id`` and
    ``recorded_at``.

    Opened in WAL mode, so a reader can query the file while a run
    is still writing to it.

    Usage::

        with SqliteCallLogger("data/calls.db") as logger:
            agent = Chatter(..., call_logger=logger)

    The database is created on the first record written, not on
    construction, so a logger that never writes leaves no file —
    the same behaviour as `JsonCallLogger`.

    Attributes:
        path (str | Path): Database file. Parent directories are
            created on first write.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def _connection(self) -> sqlite3.Connection:
        """Return the connection, opening it on first use.

        The caller holds ``self._lock``.
        """
        if self._conn is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False is safe here: every write
            # goes through __call__, which holds the lock.
            self._conn = sqlite3.connect(
                str(self._path), check_same_thread=False
            )
            # WAL lets a reader summarize the log while a run is
            # still appending to it.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_CALLS_SCHEMA)
            _add_missing_columns(self._conn)
            self._conn.commit()
        return self._conn

    def __call__(self, record: CallRecord) -> None:
        """Insert one call record as a row."""
        columns = _column_names()
        row = record.to_dict()
        values = [
            int(row[name]) if name in _BOOL_COLUMNS else row[name]
            for name in columns
        ]
        placeholders = ", ".join("?" * len(columns))
        with self._lock:
            conn = self._connection()
            conn.execute(
                f"INSERT INTO calls ({', '.join(columns)}) "
                f"VALUES ({placeholders})",
                values,
            )
            conn.commit()

    def close(self) -> None:
        """Close the connection, if one is open.

        Safe to call multiple times.
        """
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> "SqliteCallLogger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class CallLogConfig:
    """Configuration for where an agent's calls are written.

    The same shape as `MemoryConfig` and `CacheConfig`, so a
    config file declares a call log the way it already declares
    the other two stores.

    Attributes:
        backend: ``"json"`` for a JSON-lines file, ``"sqlite"``
            for a database.
        path: Where to write. Defaults to ``"calls.jsonl"`` for
            JSON and ``"calls.db"`` for SQLite.
    """

    backend: str = "json"
    path: str = field(default="", kw_only=True)

    def __post_init__(self) -> None:
        if self.backend not in ("json", "sqlite"):
            raise AgentConfigurationError(
                f"call log backend must be 'json' or 'sqlite', "
                f"got '{self.backend}'"
            )
        if not self.path:
            default = (
                "calls.db" if self.backend == "sqlite" else "calls.jsonl"
            )
            object.__setattr__(self, "path", default)


def create_call_logger(
    config: CallLogConfig,
) -> "JsonCallLogger | SqliteCallLogger":
    """Create a call-log sink from configuration.

    :param config: Call-log configuration.
    :return: A logger callable accepting one `CallRecord`.
    """
    if config.backend == "sqlite":
        return SqliteCallLogger(config.path)
    return JsonCallLogger(config.path)


def _is_sqlite(path: Path) -> bool:
    """Whether a file carries SQLite's format header.

    Sniffed rather than guessed from the suffix: a call log is
    named by whoever configured it, and reading a database as
    JSON lines fails with a decoding error that says nothing
    about the real cause.
    """
    try:
        with path.open("rb") as fh:
            return fh.read(16) == b"SQLite format 3\x00"
    except OSError:
        return False


def read_call_log(source: str | Path) -> Iterator[CallRecord]:
    """Read back the records of a call log.

    Accepts either sink's output — a JSON-lines file written by
    `JsonCallLogger` or a database written by `SqliteCallLogger`,
    told apart by the file's own header rather than its name.

    :param source: Path to the log.
    :return: An iterator of `CallRecord`, in the order written.
    :raises FileNotFoundError: If the path does not exist.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"No call log at {path}")
    if _is_sqlite(path):
        yield from _read_sqlite(path)
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield CallRecord(**json.loads(line))


def _read_sqlite(path: Path) -> Iterator[CallRecord]:
    """Yield every row of a SQLite call log, oldest first.

    Reads only the columns the file actually has, so a log
    written by a version that predates a field stays readable.
    The database is opened read-only and is never migrated here:
    reading someone's log must not require write access to it,
    and an old log kept for reference should not be rewritten by
    being looked at. Fields the file lacks take `CallRecord`'s
    defaults, which is ``None`` for anything nullable — nothing
    measured them when those rows were written.
    """
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        present = {
            row[1]
            for row in conn.execute("PRAGMA table_info(calls)").fetchall()
        }
        columns = [n for n in _column_names() if n in present]
        rows = conn.execute(
            f"SELECT {', '.join(columns)} FROM calls ORDER BY row_id"
        )
        for row in rows:
            values = dict(zip(columns, row, strict=True))
            for name in _BOOL_COLUMNS:
                if name in values:
                    values[name] = bool(values[name])
            yield CallRecord(**values)
    finally:
        conn.close()


@dataclass(frozen=True)
class CallStats:
    """Aggregate statistics over a set of call records.

    Attributes:
        calls (int): Number of records.
        successes (int): Records with ``success=True``.
        failures (int): Records with ``success=False``.
        tokens (int): Sum of completion tokens.
        prompt_tokens (int): Sum of prompt tokens.
        cached_prompt_tokens (int): Sum of cached prompt tokens.
        total_duration (float): Sum of durations in seconds.
        cost_cents (float | None): Sum of the estimated costs, or
            None when no record carried one — so an unpriced run
            reads as unpriced rather than as free.
        cached_calls (int): How many of ``calls`` were served
            from the response cache. Their tokens are counted in
            the sums above but were never billed, so a run whose
            ``cached_calls`` is large has spent far less than
            its token totals suggest.
        retries (int): Sum of the failed attempts across every
            record. Nonzero means the run was fighting the
            provider, which is the usual explanation for a run
            that took far longer than its token counts suggest.
        retry_duration (float): Seconds those failed attempts and
            their backoff waits took, on top of
            ``total_duration``.
        retried_calls (int): How many of ``calls`` needed at
            least one retry. Distinguishes one call retried five
            times from five calls retried once.
        models (tuple[str, ...]): The distinct models that
            actually answered these calls, sorted. A name cannot
            be summed, so this is the aggregate: normally one
            entry, and more than one exactly where a bucket's
            calls were not all answered by the same model — a
            ``fallback_model`` that fired partway through a run,
            or an endpoint that changed what it serves an alias.
            Empty when no record named a model.
    """

    calls: int = 0
    successes: int = 0
    failures: int = 0
    tokens: int = 0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    total_duration: float = 0.0
    cost_cents: float | None = None
    cached_calls: int = 0
    retries: int = 0
    retry_duration: float = 0.0
    retried_calls: int = 0
    models: tuple[str, ...] = ()

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
        by_run (dict[str, CallStats]): Stats keyed by ``run_id``,
            holding only the records that carried one. A log
            where nobody named a run summarizes to an empty
            mapping rather than to a single bucket of Nones.
        errors (dict[str, int]): Error message to occurrence
            count, for failed calls that carried an error.
    """

    overall: CallStats
    by_agent: dict[str, CallStats] = field(default_factory=dict)
    by_model: dict[str, CallStats] = field(default_factory=dict)
    by_run: dict[str, CallStats] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)

    def format(self) -> str:
        """Render the summary as a human-readable text block."""
        o = self.overall
        lines = [
            f"Call log summary: {o.calls} calls, "
            f"{o.successes} ok ({o.success_rate:.1%}), "
            f"{o.failures} failed",
            f"Tokens: {o.tokens} completion + {o.prompt_tokens} "
            f"prompt; total duration {o.total_duration:.2f}s "
            f"(avg {o.avg_duration:.2f}s)",
            # "unpriced" rather than 0.00: no record carried a
            # cost, which is not the same as the run being free.
            "Cost: "
            + (
                f"{o.cost_cents:.4f}c"
                if o.cost_cents is not None
                else "unpriced"
            ),
        ]
        if o.cached_calls:
            lines.append(
                f"Cache: {o.cached_calls} of {o.calls} calls served "
                f"from cache (their tokens are counted above but "
                f"were not billed)"
            )
        if o.retries:
            lines.append(
                f"Retries: {o.retries} failed attempts across "
                f"{o.retried_calls} of {o.calls} calls, costing "
                f"{o.retry_duration:.2f}s on top of the duration "
                f"above"
            )
        if self.by_agent:
            lines.append("")
            lines.append("By agent:")
            lines.extend(self._rows(self.by_agent))
        if self.by_model:
            lines.append("")
            lines.append("By model:")
            lines.extend(self._rows(self.by_model))
        if self.by_run:
            lines.append("")
            lines.append("By run:")
            lines.extend(self._rows(self.by_run))
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
    acc["cached_prompt_tokens"] += rec.get("cached_prompt_tokens") or 0
    acc["total_duration"] += rec.get("total_duration") or 0.0
    if rec.get("cached"):
        acc["cached_calls"] += 1
    retries = rec.get("retries") or 0
    acc["retries"] += retries
    acc["retry_duration"] += rec.get("retry_duration") or 0.0
    if retries:
        acc["retried_calls"] += 1
    if model := rec.get("model"):
        # Collected rather than summed: a name has no total. Two
        # entries in one bucket is the signal that its calls were
        # not all answered by the same model.
        acc["models"].add(model)
    cost = rec.get("cost_cents")
    if cost is not None:
        # Stays None until some record carries a cost, so a run on
        # an unpriced engine reports "unpriced", not "$0.00".
        acc["cost_cents"] = (acc["cost_cents"] or 0.0) + cost


def _new_acc() -> dict:
    """Return a fresh zeroed accumulator dict."""
    return {
        "calls": 0,
        "successes": 0,
        "failures": 0,
        "tokens": 0,
        "prompt_tokens": 0,
        "cached_prompt_tokens": 0,
        "total_duration": 0.0,
        "cost_cents": None,
        "cached_calls": 0,
        "retries": 0,
        "retry_duration": 0.0,
        "retried_calls": 0,
        "models": set(),
    }


def summarize_call_log(
    source: str | Path | Iterable[object],
) -> CallLogSummary:
    """Summarize a JSONL call log into aggregate statistics.

    :param source: A path to a log written by either sink — a
        JSONL file from ``JsonCallLogger`` or a database from
        ``SqliteCallLogger``, told apart by the file's own
        header — or an iterable of records where each item is a
        ``CallRecord``, a dict, or a JSON string (one JSONL
        line). Blank lines are skipped.
    :return: A :class:`CallLogSummary` with overall stats and
        per-agent / per-model / per-run breakdowns plus error
        counts.
    :raises TypeError: If an item cannot be interpreted.
    """
    if isinstance(source, (str, Path)):
        items: Iterable[object] = read_call_log(source)
    else:
        items = source

    overall = _new_acc()
    by_agent: dict[str, dict] = defaultdict(_new_acc)
    by_model: dict[str, dict] = defaultdict(_new_acc)
    by_run: dict[str, dict] = defaultdict(_new_acc)
    errors: dict[str, int] = defaultdict(int)

    for item in items:
        rec = _normalize_record(item)
        _accumulate(overall, rec)
        _accumulate(by_agent[rec.get("agent_name", "")], rec)
        _accumulate(by_model[rec.get("model_id", "")], rec)
        # Only records that named a run are bucketed, so a log
        # written before anyone did reads as having no runs
        # rather than as one run called None.
        if run_id := rec.get("run_id"):
            _accumulate(by_run[run_id], rec)
        if not rec.get("success") and rec.get("error"):
            errors[rec["error"]] += 1

    return CallLogSummary(
        overall=_finish(overall),
        by_agent={k: _finish(v) for k, v in by_agent.items()},
        by_model={k: _finish(v) for k, v in by_model.items()},
        by_run={k: _finish(v) for k, v in by_run.items()},
        errors=dict(errors),
    )


def _finish(acc: dict) -> CallStats:
    """Turn a mutable accumulator into a frozen `CallStats`.

    Only ``models`` needs converting: it accumulates as a set,
    because the aggregate of a name is the distinct names, and
    `CallStats` is frozen so it holds a sorted tuple.

    :param acc: The accumulator built by `_accumulate`.
    :return: The finished statistics.
    """
    return CallStats(**{**acc, "models": tuple(sorted(acc["models"]))})
