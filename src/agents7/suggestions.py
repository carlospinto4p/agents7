"""Suggestion queue for agent improvement.

Provides ``Suggestion`` (a frozen dataclass capturing a
proposed correction) and ``SuggestionQueue`` (a file-based
JSONL queue for persisting suggestions).

External libraries use the queue to enqueue corrections
or additions that the agent maintainer can review and
integrate into the next version.

Usage::

    from agents7.suggestions import SuggestionQueue

    queue = SuggestionQueue("tagger_agent")
    queue.add(
        input_data={"text": "moderador_sala"},
        expected={"category": "MALE"},
        note="Spanish masculine keyword",
    )

    for s in queue.list():
        print(s.input_data, s.expected, s.note)

    queue.clear()
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from agents7.exceptions import SuggestionError

_DEFAULT_DIR = Path.home() / ".agents_catalog" / "suggestions"


@dataclass(frozen=True)
class Suggestion:
    """A proposed correction for an agent.

    Attributes:
        agent (str): Agent name
            (e.g. ``"classifier_agent"``).
        input_data (dict): The input that was
            misclassified (e.g. ``{"text": "nick"}``).
        expected (dict): The expected output
            (e.g. ``{"category": "MALE"}``).
        note (str): Free-text explanation.
        timestamp (str): ISO 8601 timestamp.
    """

    agent: str
    input_data: dict
    expected: dict
    note: str
    timestamp: str

    def to_dict(self) -> dict:
        """Convert to a plain dictionary."""
        return asdict(self)


class SuggestionQueue:
    """File-based JSONL queue for agent suggestions.

    Each suggestion is appended as a JSON line. The
    file is created on first write. Default storage
    location: ``~/.agents_catalog/suggestions/<agent>.jsonl``

    Args:
        agent_name: Name of the agent this queue
            belongs to.
        path: Custom file path. If None, uses the
            default location.
    """

    def __init__(
        self,
        agent_name: str,
        path: Path | None = None,
    ) -> None:
        self._agent = agent_name
        self._path = (
            path if path is not None else _DEFAULT_DIR / f"{agent_name}.jsonl"
        )

    @property
    def path(self) -> Path:
        """Resolved file path for this queue."""
        return self._path

    def add(
        self,
        input_data: dict,
        expected: dict,
        note: str = "",
    ) -> None:
        """Append a suggestion to the queue.

        Creates the file and parent directories on
        first write.

        Raises:
            SuggestionError: If ``input_data`` is empty.
        """
        if not input_data:
            raise SuggestionError("input_data must not be empty.")
        suggestion = Suggestion(
            agent=self._agent,
            input_data=input_data,
            expected=expected,
            note=note,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(suggestion.to_dict(), ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def list(self) -> list[Suggestion]:
        """Read all suggestions from the queue.

        Returns an empty list if the file does not exist.
        """
        if not self._path.exists():
            return []
        suggestions = []
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                suggestions.append(Suggestion(**data))
        return suggestions

    def count(self) -> int:
        """Number of pending suggestions."""
        if not self._path.exists():
            return 0
        n = 0
        with self._path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    def remove(self, index: int) -> Suggestion:
        """Remove a suggestion by index.

        Returns the removed suggestion. Rewrites the
        file without the removed line.

        Raises:
            SuggestionError: If the index is out of range.
        """
        items = self.list()
        if index < 0 or index >= len(items):
            raise SuggestionError(
                f"Index {index} out of range (0–{len(items) - 1})."
            )
        removed = items.pop(index)
        if items:
            with self._path.open("w", encoding="utf-8") as f:
                for s in items:
                    line = json.dumps(s.to_dict(), ensure_ascii=False)
                    f.write(line + "\n")
        else:
            self._path.unlink(missing_ok=True)
        return removed

    def clear(self) -> None:
        """Remove all suggestions from the queue."""
        self._path.unlink(missing_ok=True)
