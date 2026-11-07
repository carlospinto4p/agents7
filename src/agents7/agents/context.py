"""Per-conversation context values for an agent's prompt.

`prompt_params` holds values fixed by the configuration. A
`ContextParam` declares a value the *caller* supplies when a
conversation starts — the nickname of the person on the other
side, the current time of day — or that a resolver function
computes fresh before every message.

A config declares them; `BaseAgent.set_context` fills them in::

    context_params:
        nick:
            default: "alguien"
            description: The other person's name
        time_of_day:
            callable: agents7.agents.context.spanish_time_of_day

Each declared name is substituted into the system prompt as
``$name``. A name nobody supplies falls back to its resolver,
then to its ``default`` — never to the literal ``$name``.

A value can also be worked out from another one. ``from`` names
the declared param it reads, and its ``callable`` then takes that
param's value as its one argument::

    context_params:
        nick:
            default: "alguien"
        nick_hint:
            callable: mypackage.hints.read_nick
            from: nick

So whoever supplies ``nick`` gets ``nick_hint`` with it, and the
two cannot fall out of step.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from agents7.exceptions import PromptParameterError
from agents7.logs import logger


@dataclass(frozen=True)
class ContextParam:
    """A prompt value supplied per conversation.

    Attributes:
        name (str): The placeholder name, substituted as
            `$name` in the system prompt.
        default (str): Value used when neither the caller nor
            a resolver provides one. Defaults to empty string.
        description (str): What the value means, shown to
            whoever supplies it (e.g. the UI's chat form).
            Defaults to empty string.
        resolver (Callable, optional): Zero-argument callable
            returning the value, called before every message.
            A caller-supplied value takes precedence over it.
            With `derived_from` it takes one argument instead.
            Defaults to None.
        derived_from (str): Name of another declared param
            this one is computed from. The resolver is then
            called with that param's current value. Requires
            a resolver. Defaults to empty string.
    """

    name: str
    default: str = ""
    description: str = ""
    resolver: Callable[..., str] | None = None
    derived_from: str = ""

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise PromptParameterError(
                "`ContextParam.name` must be a non-empty string."
            )
        if not isinstance(self.default, str):
            raise PromptParameterError(
                f"context_params['{self.name}'].default must be a "
                f"str, got {type(self.default).__name__}."
            )
        if self.resolver is not None and not callable(self.resolver):
            raise PromptParameterError(
                f"context_params['{self.name}'].callable must be "
                f"a callable, got {type(self.resolver).__name__}."
            )
        if not isinstance(self.derived_from, str):
            raise PromptParameterError(
                f"context_params['{self.name}'].from must be a "
                f"str, got {type(self.derived_from).__name__}."
            )
        if self.derived_from and self.resolver is None:
            raise PromptParameterError(
                f"context_params['{self.name}'] is derived from "
                f"`{self.derived_from}` but declares no callable "
                f"to compute it with."
            )
        if self.derived_from == self.name:
            raise PromptParameterError(
                f"context_params['{self.name}'] cannot be derived "
                f"from itself."
            )

    @property
    def identity(self) -> str:
        """This declaration, as it enters a run's fingerprint.

        The *declaration* — name, default and where a resolver
        comes from — never the resolved value. A resolver reads
        the clock and a caller-supplied value is input rather
        than configuration, so either one in the fingerprint
        would split a single agent's measurement history into a
        band per hour of the day. The same reasoning that keeps
        a drawn `prompt_variants` text out of it.

        :return: A stable line identifying this declaration.
        """
        if self.resolver is None:
            source = ""
        else:
            module = getattr(self.resolver, "__module__", "")
            qualname = getattr(
                self.resolver,
                "__qualname__",
                type(self.resolver).__name__,
            )
            source = f"{module}.{qualname}"
        identity = f"{self.name}|{self.default}|{source}"
        if self.derived_from:
            identity += f"|from:{self.derived_from}"
        return identity

    def resolve(self, source_value: str = "") -> str:
        """Compute the value to substitute.

        Calls the resolver when one is declared, falling back
        to `default` if it raises.

        :param source_value: Current value of the `derived_from`
            param, passed to the resolver. Ignored when this
            param is not derived. Defaults to empty string.
        :return: The resolved value.
        """
        if self.resolver is None:
            return self.default
        try:
            if self.derived_from:
                return str(self.resolver(source_value))
            return str(self.resolver())
        except Exception as exc:
            logger.warning(
                f"context_params['{self.name}']: resolver raised "
                f"{type(exc).__name__}: {exc}. Using the default."
            )
            return self.default


def _hour_bucket(hour: int) -> str:
    """Map a 24-hour clock hour to a coarse part of the day.

    :param hour: Hour of the day, 0-23.
    :return: One of ``night``, ``morning``, ``afternoon``,
        ``evening``.
    """
    if hour < 6:
        return "night"
    if hour < 13:
        return "morning"
    if hour < 21:
        return "afternoon"
    return "evening"


def time_of_day() -> str:
    """Current part of the day in English, from the local clock.

    Declare it in a config as
    ``callable: agents7.agents.context.time_of_day``.

    :return: ``night``, ``morning``, ``afternoon`` or
        ``evening``.
    """
    return _hour_bucket(datetime.now().hour)


_SPANISH_TIME_OF_DAY = {
    "night": "de madrugada",
    "morning": "por la mañana",
    "afternoon": "por la tarde",
    "evening": "por la noche",
}


def spanish_time_of_day() -> str:
    """Current part of the day in Spanish, from the local clock.

    Declare it in a config as
    ``callable: agents7.agents.context.spanish_time_of_day``.

    :return: ``de madrugada``, ``por la mañana``, ``por la
        tarde`` or ``por la noche``.
    """
    return _SPANISH_TIME_OF_DAY[time_of_day()]


# Indexed by `datetime.weekday()`, where Monday is 0. Written out
# rather than taken from `strftime("%A")`, which follows the
# process locale and so answers in whatever language the machine
# happens to be set to.
_SPANISH_WEEKDAYS = (
    "lunes",
    "martes",
    "miércoles",
    "jueves",
    "viernes",
    "sábado",
    "domingo",
)


def spanish_weekday() -> str:
    """Current day of the week in Spanish, from the local clock.

    Declare it in a config as
    ``callable: agents7.agents.context.spanish_weekday``.

    :return: ``lunes``, ``martes``, ``miércoles``, ``jueves``,
        ``viernes``, ``sábado`` or ``domingo``.
    """
    return _SPANISH_WEEKDAYS[datetime.now().weekday()]
