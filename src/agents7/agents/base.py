import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from string import Template
from typing import Any, TypeVar

from agents7.agents.context import ContextParam
from agents7.exceptions import PromptParameterError
from agents7.llm import LLMEngine
from agents7.logs import logger
from agents7.models import AgentResponse

Agent = TypeVar("Agent", bound="BaseAgent")


@dataclass(frozen=True)
class BaseAgent(ABC):
    """
    Base class for all agents.
    This class should be inherited by all agent classes.

    Attributes:
        name (str): The name of the agent.
        version (int): The version of the agent.
        engine (LLMEngine): The LLM engine used by the agent.
        system_prompt (str): The system prompt for the agent.
        prompt_params (dict, optional): Dictionary of parameters to
            substitute into the system_prompt using
            $variable placeholders. Values can be strings or
            callables returning strings. Defaults to None.
        context_params (tuple, optional): `ContextParam`
            declarations for values supplied per conversation
            rather than fixed by the config — see `set_context`.
            Defaults to None.
        prompt_variants (dict, optional): Placeholder name to two
            or more alternative texts. One is drawn for each name
            when the agent is built — see `draw_variants`.
            Defaults to None.
    """

    name: str
    version: int
    engine: LLMEngine
    system_prompt: str
    prompt_params: dict[str, str | Callable[[], str]] | None = field(
        default=None, kw_only=True
    )
    context_params: tuple[ContextParam, ...] | None = field(
        default=None, kw_only=True
    )
    prompt_variants: dict[str, tuple[str, ...]] | None = field(
        default=None, kw_only=True
    )

    def __post_init__(self) -> None:
        if self.prompt_params is not None:
            if not isinstance(self.prompt_params, dict):
                raise PromptParameterError(
                    "`prompt_params` must be a dict or None."
                )
            for key, value in self.prompt_params.items():
                if not isinstance(value, str) and not callable(value):
                    raise PromptParameterError(
                        f"prompt_params['{key}'] must be a str "
                        f"or callable, got {type(value).__name__}."
                    )
        self._validate_context_params()
        self._validate_prompt_variants()
        self.draw_variants()

    def _validate_prompt_variants(self) -> None:
        """Check the `prompt_variants` declarations are usable.

        One text is not a variant, and a text listed twice is drawn
        twice as often without anyone having said so, so both are
        errors rather than something to tolerate.
        """
        if self.prompt_variants is None:
            return
        if not isinstance(self.prompt_variants, dict):
            raise PromptParameterError(
                "`prompt_variants` must be a dict of name to texts."
            )
        normalised: dict[str, tuple[str, ...]] = {}
        for name, texts in self.prompt_variants.items():
            if isinstance(texts, str) or not isinstance(texts, (tuple, list)):
                raise PromptParameterError(
                    f"prompt_variants['{name}'] must be a list of texts."
                )
            if len(texts) < 2:
                raise PromptParameterError(
                    f"prompt_variants['{name}'] needs at least two "
                    f"texts to choose between, got {len(texts)}."
                )
            if not all(isinstance(t, str) and t.strip() for t in texts):
                raise PromptParameterError(
                    f"prompt_variants['{name}'] holds a text that is "
                    f"empty or not a string."
                )
            if len(set(texts)) != len(texts):
                raise PromptParameterError(
                    f"prompt_variants['{name}'] lists the same text twice."
                )
            normalised[name] = tuple(texts)
        self._set_attr("prompt_variants", normalised)
        others = set(self.prompt_params or {}) | {
            p.name for p in self.context_params or ()
        }
        clashes = set(normalised) & others
        if clashes:
            raise PromptParameterError(
                f"`{', '.join(sorted(clashes))}` is declared in "
                f"prompt_variants and in prompt_params or "
                f"context_params. A name belongs to one of them."
            )

    @property
    def variant_values(self) -> dict[str, str]:
        """The text drawn for each `prompt_variants` name."""
        return dict(getattr(self, "_variant_values", {}))

    def draw_variants(self, seed: str | None = None) -> None:
        """Pick one text for each `prompt_variants` name.

        Called when the agent is built, so a new agent is a new
        draw. Several alternatives written into a fixed prompt do
        not vary: measured on one agent, 16th September 2026, the
        model answered with whichever came first in 8, 9 and 10
        conversations of 10. Showing one, chosen per conversation,
        is what makes two conversations differ.

        :param seed: Makes the draw repeatable — the same seed
            picks the same texts in any process. None draws
            afresh.
        """
        if not self.prompt_variants:
            return
        rng = random.Random(seed)
        self._set_attr(
            "_variant_values",
            {
                name: rng.choice(texts)
                for name, texts in sorted(self.prompt_variants.items())
            },
        )

    def _validate_context_params(self) -> None:
        """Check the `context_params` declarations are usable."""
        if self.context_params is None:
            return
        if isinstance(self.context_params, ContextParam) or not isinstance(
            self.context_params, (tuple, list)
        ):
            raise PromptParameterError(
                "`context_params` must be a tuple of ContextParam."
            )
        self._set_attr("context_params", tuple(self.context_params))
        seen: set[str] = set()
        for param in self.context_params:
            if not isinstance(param, ContextParam):
                raise PromptParameterError(
                    f"`context_params` entries must be ContextParam, "
                    f"got {type(param).__name__}."
                )
            if param.name in seen:
                raise PromptParameterError(
                    f"context_params declares `{param.name}` twice."
                )
            seen.add(param.name)
        by_name = {p.name: p for p in self.context_params}
        for param in self.context_params:
            if not param.derived_from:
                continue
            source = by_name.get(param.derived_from)
            if source is None:
                raise PromptParameterError(
                    f"context_params['{param.name}'] is derived from "
                    f"`{param.derived_from}`, which is not declared."
                )
            if source.derived_from:
                raise PromptParameterError(
                    f"context_params['{param.name}'] is derived from "
                    f"`{source.name}`, which is itself derived. A "
                    f"derived param reads a plain one."
                )
        clashes = seen & set(self.prompt_params or {})
        if clashes:
            raise PromptParameterError(
                f"`{', '.join(sorted(clashes))}` is declared in both "
                f"prompt_params and context_params. A name belongs to "
                f"one or the other."
            )

    @property
    def context_values(self) -> dict[str, str]:
        """Values supplied for this conversation via `set_context`."""
        return dict(getattr(self, "_context_values", {}))

    @property
    def resolved_context(self) -> dict[str, str]:
        """What the `context_params` resolve to, right now.

        The values the model actually reads: each declared
        param resolved, then overlaid with anything
        `set_context` supplied. `_substitute` uses this same
        mapping, so the two cannot drift.

        A param declared with `derived_from` is computed last,
        from the value its source ended up with — supplied or
        not — unless the caller supplied the derived one too.

        Exposed because it is otherwise unrecoverable. A
        resolver runs inside substitution and its result is
        discarded, so nothing afterwards can say which time of
        day, which nickname or which hint a finished
        conversation was run with -- and at temperature a
        conversation cannot be run again to find out. A reviewer
        reading a transcript needs it, and
        `eval.chat_review.ConversationRecorder` records it.

        **Reading it twice can give different values**, because
        a `callable` resolver is called each time. Capture it
        once, at the point the answer is wanted.

        :return: Declared context names and their current values.
        """
        params = self.context_params or ()
        supplied = self.context_values
        resolved = {p.name: p.resolve() for p in params if not p.derived_from}
        resolved.update(supplied)
        for param in params:
            if param.derived_from and param.name not in supplied:
                resolved[param.name] = param.resolve(
                    resolved[param.derived_from]
                )
        return resolved

    def set_context(self: Agent, **values: str) -> Agent:
        """Supply values for the declared `context_params`.

        Call it before the conversation's first message. Each
        keyword must match a declared `ContextParam` name;
        anything else is a typo the agent cannot use, so it
        raises rather than being silently dropped. Values
        replace whatever a resolver or `default` would give,
        and persist for the life of this instance.

        :param values: Declared context names and their values.
        :return: This same agent, so the call can be chained.
        :raises PromptParameterError: If a name is not declared.
        """
        declared = {p.name for p in self.context_params or ()}
        unknown = sorted(set(values) - declared)
        if unknown:
            raise PromptParameterError(
                f"Agent `{self.name}` does not declare the context "
                f"param(s) `{', '.join(unknown)}`. Declared: "
                f"`{', '.join(sorted(declared)) or 'none'}`."
            )
        merged = {**self.context_values}
        merged.update({k: str(v) for k, v in values.items()})
        self._set_attr("_context_values", merged)
        return self

    def substitute(self, text: str) -> str:
        """Substitute this agent's placeholders into *text*.

        The same mapping `resolved_system_prompt` uses, exposed
        so text that joins the prompt somewhere other than the
        prompt field itself — a Chatter's `opening_instructions`,
        which are appended per call — resolves identically. A
        placeholder that resolves in one of them and not the
        other would be worse than none.

        :param text: Text possibly holding ``$name`` placeholders.
        :return: The text with every declared placeholder filled.
        """
        return self._substitute(text, with_variants=True)

    def _substitute(
        self,
        text: str,
        *,
        with_variants: bool,
        with_context: bool = True,
    ) -> str:
        """`substitute`, optionally leaving some names unfilled.

        :param text: Text possibly holding ``$name`` placeholders.
        :param with_variants: Fill the `prompt_variants` names.
        :param with_context: Fill the `context_params` names.
            False leaves them as placeholders, which is what
            `system_prompt_identity` wants. Defaults to True.
        :return: The text with the chosen placeholders filled.
        """
        if not text or not (
            self.prompt_params or self.context_params or self.prompt_variants
        ):
            return text
        resolved: dict[str, str] = {}
        if with_context:
            resolved.update(self.resolved_context)
        resolved.update(
            {
                k: v() if callable(v) else v
                for k, v in (self.prompt_params or {}).items()
            }
        )
        if with_variants:
            resolved.update(self.variant_values)
        return Template(text).safe_substitute(resolved)

    @property
    def resolved_system_prompt(self) -> str:
        """Resolved system prompt with placeholders substituted."""
        return self.substitute(self.system_prompt)

    @property
    def system_prompt_identity(self) -> str:
        """The system prompt as configured, whatever was drawn.

        `resolved_system_prompt` with everything the config does
        not fix left as a placeholder and declared after it, so
        two agents built from one config share it. Two kinds of
        value qualify, for the same reason:

        - a `prompt_variants` name, which each agent draws for
          itself, listed as every alternative it could draw;
        - a `context_params` name, whose value a resolver reads
          off the clock or a caller supplies per conversation,
          listed as its declaration.

        Both would otherwise make one agent hash differently from
        one run to the next, splitting its measurement history
        into bands that describe the same configuration — a
        resolver on the time of day into one band per part of the
        day. With neither, this is exactly
        `resolved_system_prompt`.
        """
        prompt = self._substitute(
            self.system_prompt, with_variants=False, with_context=False
        )
        for param in sorted(self.context_params or (), key=lambda p: p.name):
            prompt += f"\ncontext|{param.identity}"
        for name, texts in sorted((self.prompt_variants or {}).items()):
            prompt += f"\nvariant|{name}|" + "|".join(texts)
        return prompt

    @classmethod
    @abstractmethod
    def from_yaml(cls, file_path: str) -> Agent:
        """
        Create an instance of the agent from a YAML file configuration.
        This method should be overridden by subclasses to provide specific
        YAML loading logic.

        :param file_path: The path to the YAML file containing the agent
            configuration.
        :return: An instance of the agent.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def act(self, *args: Any, **kwargs: Any) -> AgentResponse:
        """
        Perform an action based on the current state and input.
        This method should be overridden by subclasses.

        :param args: Positional arguments for the action.
        :param kwargs: Keyword arguments for the action.
        :return: An AgentResponse model, containing:
            - response: The generated text response.
            - model: The structured output validated by the output model.
            - thinking: The thinking part of the response, if any.
            - metrics: LLMMetrics model, containing:
                - tokens: The number of tokens in the response.
                - prompt_tokens: The number of tokens in the prompt.
                - total_duration: The total duration of the response,
                 in seconds.
            - idx: Optional index for tracking purposes.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def with_overrides(self: Agent, **kwargs: Any) -> Agent:
        """Create a copy of this agent with selected fields
        replaced.

        Useful for A/B testing, model comparison, or creating
        agent variants without rebuilding from scratch.

        Accepts any field defined on the concrete agent class. The
        engine can be replaced by passing a new `LLMEngine`.

        :param kwargs: Fields to override. For example:
            ``agent.with_overrides(system_prompt="New prompt")``
            ``agent.with_overrides(temperature=0.0)``
        :return: A new agent instance with the overridden fields.

        Values supplied through `set_context` come along, for every
        name the copy still declares. They are not a dataclass
        field, so a bare `dataclasses.replace` drops them and the
        copy answers with each param's default: on 20th September
        2026 a whole read of one agent ran that way, and the agent
        was judged for ignoring a nickname it had never been sent.
        """
        copy = replace(self, **kwargs)
        declared = {p.name for p in copy.context_params or ()}
        supplied = {
            name: value
            for name, value in self.context_values.items()
            if name in declared
        }
        if supplied:
            copy._set_attr("_context_values", supplied)
        return copy

    def _set_attr(self, name: str, value) -> None:
        """Mutate a field on a frozen dataclass instance.

        **Only for fields read at call time**, such as `engine` or
        `guard_callback`. Anything `__post_init__` folds into
        another field is already baked in, and setting it here
        changes nothing: a `Detector` builds the text it sends
        into `_prompt` once, so a guard whose `system_prompt` is
        replaced this way keeps judging by the old one. On 17th
        September 2026 that silently produced a whole A/B
        comparison in which both arms were the shipped prompt —
        replacing it with "approve everything" changed no verdict.
        Use `with_overrides`, which rebuilds the agent.
        """
        object.__setattr__(self, name, value)

    async def aclose(self) -> None:
        """Close the async clients of every engine this agent holds.

        Its own engine, and those of any agent it carries in a
        field — a chatter's guard and fact extractor included — each
        closed once. Call it before the event loop that ran the
        agent ends; see `LLMEngine.aclose`.
        """
        for engine in self._engines():
            await engine.aclose()

    def _engines(self) -> list[LLMEngine]:
        """Every distinct engine reachable through agent fields."""
        engines: dict[int, LLMEngine] = {}
        pending: list[BaseAgent] = [self]
        visited: set[int] = set()
        while pending:
            agent = pending.pop()
            if id(agent) in visited:
                continue
            visited.add(id(agent))
            for f in fields(agent):
                value = getattr(agent, f.name, None)
                if isinstance(value, LLMEngine):
                    engines.setdefault(id(value), value)
                elif isinstance(value, BaseAgent):
                    pending.append(value)
        return list(engines.values())

    def warm_up(self) -> None:
        """Pre-load the LLM model by sending a trivial prompt.

        Ollama loads models lazily on first call, which can
        cause a multi-second cold-start delay. Call this
        method at application startup to pay that cost
        upfront.

        For non-Ollama providers this is a no-op.
        """
        if not self.engine.supports_warmup():
            return
        self._log_warm_up_start()
        self.engine.generate(prompt="ping", max_tokens=1)
        self._log_warm_up_done()

    async def warm_up_async(self) -> None:
        """Async version of `warm_up`."""
        if not self.engine.supports_warmup():
            return
        self._log_warm_up_start()
        await self.engine.generate_async(
            prompt="ping",
            max_tokens=1,
        )
        self._log_warm_up_done()

    def _log_warm_up_start(self) -> None:
        """Log warm-up start."""
        logger.info(
            f"Agent `{self.name}`: warming up model "
            f"`{self.engine.model_name}`..."
        )

    def _log_warm_up_done(self) -> None:
        """Log warm-up completion."""
        logger.info(
            f"Agent `{self.name}`: model `{self.engine.model_name}` ready."
        )
