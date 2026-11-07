"""Character facts that outlive a single conversation.

`MemoryBackend` persists a *transcript* keyed by
``conversation_id``, so anything a character improvises about
itself is lost the moment the next conversation starts. This
module stores facts keyed by the *character* instead, folds
them back into the system prompt on every turn, and extracts
new ones from a finished transcript.

Facts are stored in the character's own language, so a fact
folded back into the prompt reads as something that character
would say rather than a translation of it.

Usage::

    from agents7.agents.facts import (
        FactExtractor,
        JsonFileFactStore,
    )

    store = JsonFileFactStore("facts/")
    chatter = Chatter(
        ...,
        facts=store,
        character_id="david",
        fact_extractor=FactExtractor(
            name="david-facts", version=1, engine=engine
        ),
    )
    chatter.chat("¿Cuál es tu película favorita?")
    chatter.remember()   # writes what he said into the store
"""

import json
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from agents7.agents.labeler import Labeler


@dataclass(frozen=True)
class CharacterFact:
    """One durable thing that is true of a character.

    Attributes:
        text: The fact itself, as a short standalone
            sentence in the character's own language
            (e.g. ``"Su película favorita es El Club de la
            Lucha"``).
        source_conversation: The ``conversation_id`` the
            fact was extracted from, or ``""`` when it was
            added by hand.
    """

    text: str
    source_conversation: str = ""

    def to_dict(self) -> dict[str, str]:
        """Render as a JSON-serialisable dict."""
        return {
            "text": self.text,
            "source_conversation": self.source_conversation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "CharacterFact":
        """Build a fact from a stored dict.

        :param data: Mapping with a ``text`` key and an
            optional ``source_conversation`` key.
        :return: A `CharacterFact` instance.
        """
        return cls(
            text=str(data["text"]),
            source_conversation=str(data.get("source_conversation", "")),
        )


def _normalise(text: str) -> str:
    """Collapse a fact to a comparable form for de-duplication."""
    return " ".join(text.lower().split()).rstrip(".!?")


class FactStore(ABC):
    """Abstract base class for character fact storage."""

    @abstractmethod
    def load(self, character_id: str) -> list[CharacterFact]:
        """Load every recorded fact for a character.

        :param character_id: Character identifier.
        :return: List of facts, oldest first. Empty when the
            character has none.
        """

    @abstractmethod
    def add(
        self,
        character_id: str,
        facts: Iterable[CharacterFact | str],
    ) -> list[CharacterFact]:
        """Append facts, skipping ones already recorded.

        Comparison ignores case, surrounding whitespace, and a
        trailing full stop, so re-running an extraction over
        the same conversation adds nothing.

        :param character_id: Character identifier.
        :param facts: Facts to append. Plain strings are
            accepted and stored with no source conversation.
        :return: The facts that were actually written, in the
            order given. Empty when all were already known.
        """

    @abstractmethod
    def clear(self, character_id: str) -> None:
        """Remove every fact for a character.

        :param character_id: Character identifier.
        """

    @abstractmethod
    def list_characters(self) -> list[str]:
        """List all character IDs with recorded facts.

        :return: Sorted list of character ID strings.
        """


class JsonFileFactStore(FactStore):
    """JSON file-backed character facts.

    Stores each character's facts as a separate JSON file in
    a directory::

        facts/
            david.json    # [{"text": "...", ...}, ...]
            marta.json

    JSON is the only backend on purpose: the files are meant
    to be opened and hand-corrected when an extraction records
    something wrong.

    Thread-safe via a lock. Creates the directory on first
    write.
    """

    def __init__(self, path: str = "facts/") -> None:
        self._dir = Path(path)
        self._lock = threading.Lock()

    def load(self, character_id: str) -> list[CharacterFact]:
        """Load a character's facts from its JSON file."""
        with self._lock:
            return self._read(character_id)

    def add(
        self,
        character_id: str,
        facts: Iterable[CharacterFact | str],
    ) -> list[CharacterFact]:
        """Append new facts to a character's JSON file."""
        incoming = [
            f if isinstance(f, CharacterFact) else CharacterFact(text=str(f))
            for f in facts
        ]
        incoming = [f for f in incoming if f.text.strip()]
        if not incoming:
            return []
        with self._lock:
            existing = self._read(character_id)
            seen = {_normalise(f.text) for f in existing}
            added: list[CharacterFact] = []
            for fact in incoming:
                key = _normalise(fact.text)
                if key in seen:
                    continue
                seen.add(key)
                added.append(fact)
            if not added:
                return []
            self._write(character_id, existing + added)
            return added

    def clear(self, character_id: str) -> None:
        """Remove a character's JSON file."""
        with self._lock:
            self._path_for(character_id).unlink(missing_ok=True)

    def list_characters(self) -> list[str]:
        """List all character IDs with a stored file."""
        with self._lock:
            if not self._dir.exists():
                return []
            return sorted(p.stem for p in self._dir.glob("*.json"))

    # --- Private helpers ---

    def _read(self, character_id: str) -> list[CharacterFact]:
        """Read a character's facts, assuming the lock is held."""
        path = self._path_for(character_id)
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as f:
            raw = json.load(f)
        return [CharacterFact.from_dict(item) for item in raw]

    def _write(
        self,
        character_id: str,
        facts: list[CharacterFact],
    ) -> None:
        """Write a character's facts, assuming the lock is held."""
        self._dir.mkdir(parents=True, exist_ok=True)
        with self._path_for(character_id).open("w", encoding="utf-8") as f:
            json.dump(
                [fact.to_dict() for fact in facts],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def _path_for(self, character_id: str) -> Path:
        """Return the file path for a character."""
        allowed = frozenset(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789-_."
        )
        safe = "".join(c if c in allowed else "_" for c in character_id)
        # Guard against identifiers that are empty or consist solely
        # of dots (e.g. "." or ".."), which are special directory
        # entries.
        if not safe.strip("."):
            safe = "_"
        path = self._dir / f"{safe}.json"
        # Belt-and-suspenders: result must be a direct child of _dir.
        if path.parent != self._dir:
            raise ValueError(
                f"Unsafe character_id produced an "
                f"out-of-directory path: `{path}`"
            )
        return path


class ExtractedFacts(BaseModel):
    """Output model for `FactExtractor`."""

    facts: list[str] = Field(
        default_factory=list,
        description=(
            "New durable facts the character stated about "
            "itself. Empty when the conversation carries none."
        ),
    )


_EXTRACTOR_PROMPT = """\
You read a finished conversation and write down what the \
character said about ITSELF that should still be true the \
next time it talks to somebody else.

## Your Task
You will receive a JSON object with:
- ``character``: the character's name.
- ``language``: the language the character speaks, and the \
language to write the facts in. When it is empty, follow the \
language the character speaks in the transcript.
- ``known_facts``: what is already recorded about it.
- ``transcript``: the conversation. ``assistant`` turns are \
the character speaking; ``user`` turns are the other person.

Return the new facts.

## What Counts As A Fact
Record something only when ALL of these hold:
- The CHARACTER said it about itself, in an ``assistant`` \
turn. Never record what the other person said about \
themselves.
- It is durable — a preference, a possession, a habit, \
somebody in the character's life, something that happened \
to it. It would still be true in a month.
- It is new: not in ``known_facts``, and not a restatement \
of one.

## What To Leave Out
- Anything about the other person.
- Passing states: what the character is doing, feeling or \
has just done at the moment it speaks. The test is whether it \
stops being true once the conversation ends. These are all \
passing states, not facts: "is busy with the computer", "is \
waiting for a build to finish", "is tired today", "laughed at \
the first question", "has just had dinner".
- Greetings, questions, and jokes with no content.
- Anything the character only asked about, guessed at, or \
denied.

## How To Write Each Fact
- One short sentence, in ``language``. Write every fact in \
that one language even when the other person wrote in \
another — translate it rather than switching language \
part-way.
- Keep proper nouns exactly as the character wrote them: \
names of people, places, films, songs, teams, brands. A \
character who said "El Club de la Lucha" keeps that title; \
do not render it as "Fight Club".
- Standalone — it must make sense with no conversation \
around it. For a Spanish-speaking character, write "Su \
película favorita es El Club de la Lucha", not "Le gusta esa \
película".
- In the third person, about the character.
- Keep the character's own tense, and keep the words that \
place it in time. Something it put in the past stays in the \
past: a character who said "la de mi piso en Bilbao, hace \
años" gets "Tuvo un piso en Bilbao hace años", never "Tiene \
un piso en Bilbao". Never promote a past fact into a \
present one, and never drop a "hace años", "de pequeño" or \
"antes" that was there.
- No hedging and no reporting verbs: nothing that means \
"seems to" or "the character said that".

## Returning Nothing
Most short conversations carry no fact at all, and an empty \
list is the right answer for them — a normal outcome, not a \
failure. Never soften a passing state into a fact to avoid \
coming back empty-handed. A wrong fact is folded into every \
later conversation the character has; an empty list costs \
nothing.\
"""


@dataclass(frozen=True)
class FactExtractor(Labeler):
    """Reads a transcript and returns new character facts.

    A `Labeler` with a fixed prompt and output model, so it
    needs only a name, a version, and an engine::

        extractor = FactExtractor(
            name="david-facts", version=1, engine=engine
        )

    Call it through `Chatter.remember()` rather than
    directly; that method supplies the transcript and the
    already-known facts, and writes the result to the store.
    """

    system_prompt: str = field(default=_EXTRACTOR_PROMPT)
    output_model: type[BaseModel] = field(
        default=ExtractedFacts,
        kw_only=True,
    )

    def extract(
        self,
        character: str,
        transcript: list[dict[str, str]],
        known_facts: list[str] | None = None,
        language: str = "",
    ) -> list[str]:
        """Extract new facts from a finished transcript.

        :param character: The character's name, for the prompt.
        :param transcript: Conversation history as ``role`` /
            ``content`` dicts.
        :param known_facts: Facts already recorded, so the
            model does not repeat them.
        :param language: The character's language, which the
            facts are written in (e.g. ``"Spanish"``).
            Defaults to ``""``, which leaves the model to
            follow the transcript. `Chatter.remember` fills
            it from the persona.
        :return: New fact sentences. Empty when the
            conversation carries none, or when the model's
            output could not be parsed.
        """
        response = self.act(
            self._build_input(character, transcript, known_facts, language),
        )
        return self._facts_from(response)

    async def extract_async(
        self,
        character: str,
        transcript: list[dict[str, str]],
        known_facts: list[str] | None = None,
        language: str = "",
    ) -> list[str]:
        """Async version of `extract`.

        :param character: The character's name, for the prompt.
        :param transcript: Conversation history as ``role`` /
            ``content`` dicts.
        :param known_facts: Facts already recorded.
        :param language: The character's language, which the
            facts are written in.
        :return: New fact sentences.
        """
        response = await self.act_async(
            self._build_input(character, transcript, known_facts, language),
        )
        return self._facts_from(response)

    @staticmethod
    def _build_input(
        character: str,
        transcript: list[dict[str, str]],
        known_facts: list[str] | None,
        language: str = "",
    ) -> dict:
        """Build the input payload for the extraction call."""
        return {
            "character": character,
            "language": language,
            "known_facts": list(known_facts or []),
            "transcript": [
                {"role": m["role"], "content": m["content"]}
                for m in transcript
            ],
        }

    @staticmethod
    def _facts_from(response) -> list[str]:
        """Pull the fact list out of an `AgentResponse`."""
        if response.model is None:
            return []
        return [text for text in response.model.facts if text.strip()]
