from enum import StrEnum
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel, Field, model_validator

from agents7.exceptions import PropertyConfigurationError


T = TypeVar("T")


class LLMMetrics(BaseModel):
    """
    LLM metrics class.
    """

    tokens: int = Field(description="The number of tokens in the response.")
    prompt_tokens: int = Field(
        description="The number of tokens in the prompt."
    )
    cached_prompt_tokens: int = Field(
        default=0,
        description=(
            "How many of `prompt_tokens` the provider served from its "
            "own prompt cache, and so billed at its cheaper cached "
            "rate. 0 when the provider does not report it or does not "
            "cache prompts."
        ),
    )
    total_duration: float = Field(
        description="The total duration of the response, in seconds."
    )
    cost_cents: float | None = Field(
        default=None,
        description=(
            "Estimated cost in cents, or None if cost tracking is "
            "not configured."
        ),
    )
    cached: bool = Field(
        default=False,
        description=(
            "Whether this response was served from the response "
            "cache rather than by calling the provider. A cache "
            "hit reports its own duration and a cost of zero, but "
            "keeps the token counts of the call that filled the "
            "cache — this flag is what tells a consumer those "
            "tokens were not billed again. On a response that "
            "aggregates several calls (a tool loop, a guarded "
            "turn) it is True only when every one of them was a "
            "hit."
        ),
    )
    model: str | None = Field(
        default=None,
        description=(
            "The `provider::model` that actually produced this "
            "response, as the provider itself named it in its "
            "reply — which is not always what the config asked "
            "for. Two ways they differ: a provider may resolve an "
            "alias (`deepseek-chat` is served by "
            "`deepseek-v4-flash`), and an engine with a "
            "`fallback_model` answers from the fallback when the "
            "primary fails. Both used to be invisible: a run "
            "reported every call against the configured name. "
            "On a response that aggregates several calls (a "
            "guarded turn, a tool loop) it holds only when every "
            "one of them named the same model, the way `cached` "
            "does — a guard usually runs a different model, and "
            "one field cannot honestly report two. The per-call "
            "truth is in the call log, one row per call. `None` "
            "where a provider named nothing back, or where an "
            "aggregate spans more than one model; never as a "
            "stand-in for the configured value."
        ),
    )
    retries: int = Field(
        default=0,
        description=(
            "How many provider calls made for this response failed "
            "and were retried. 0 is the ordinary case: every call "
            "succeeded first time. A failed attempt raises before "
            "the provider reports any usage, so what it cost in "
            "money cannot be measured — that it happened, and the "
            "time it took, can. It counts retries rather than "
            "attempts so that it still adds up on a response that "
            "aggregates several calls, where a total attempt count "
            "could not be told apart from several clean calls."
        ),
    )
    retry_duration: float = Field(
        default=0.0,
        description=(
            "Seconds spent on attempts that failed, including the "
            "backoff wait after each. Not part of `total_duration`, "
            "which is the successful call's own figure — so the "
            "wall clock a caller waited is the two added together."
        ),
    )

    @classmethod
    def sum(cls, parts: list["LLMMetrics"]) -> "LLMMetrics":
        """Add up several calls' metrics into one.

        The single aggregator, so a field added to the declaration
        cannot be forgotten by one of the places that combine
        calls — a guarded turn's reply plus its regenerations, a
        tool loop's model turns, a pipeline's steps, the UI's
        guard summary. Each of those summed by hand until this
        existed, and each would have had to be found again.

        Counters and durations add. `cost_cents` adds only when at
        least one part carries it, and stays ``None`` when no
        engine declares pricing — so an unpriced response reports
        ``None`` rather than a misleading ``0.0``. `cached` holds
        only when every part was a cache hit: one live call means
        the response cost real time and real money. `model` holds
        on the same condition — the one model every part used, or
        ``None`` when they disagree, since a guarded turn's reply
        and its guard usually run different models and one field
        cannot report both.

        :param parts: The calls' metrics, in the order they were
            made. An empty list gives an all-zero, unpriced,
            uncached total.
        :return: Their total.
        """
        costs = [p.cost_cents for p in parts if p.cost_cents is not None]
        models = {p.model for p in parts}
        return cls(
            tokens=sum(p.tokens for p in parts),
            prompt_tokens=sum(p.prompt_tokens for p in parts),
            cached_prompt_tokens=sum(p.cached_prompt_tokens for p in parts),
            total_duration=sum(p.total_duration for p in parts),
            cost_cents=sum(costs) if costs else None,
            cached=bool(parts) and all(p.cached for p in parts),
            model=models.pop() if len(models) == 1 else None,
            retries=sum(p.retries for p in parts),
            retry_duration=sum(p.retry_duration for p in parts),
        )


class AgentResponse(BaseModel):
    """
    Agent response class.
    """

    response: str = Field(description="The generated text response.")
    model: BaseModel | None = Field(
        description="The structured output validated by the output model."
    )
    thinking: str | None = Field(
        default=None, description="The thinking part of the response, if any."
    )
    metrics: LLMMetrics = Field(description="Metrics of the LLM request.")
    idx: int | None = Field(
        default=None, description="Index for tracking purposes."
    )


class ChatResponse(BaseModel):
    """Response from a conversational Chatter agent.

    Unlike `AgentResponse`, this has no structured output
    model — just the raw text reply, optional thinking
    tokens, and LLM metrics.
    """

    response: str = Field(description="The assistant's text reply.")
    thinking: str | None = Field(
        default=None,
        description="The thinking part of the response, if any.",
    )
    metrics: LLMMetrics = Field(description="Metrics of the LLM request.")


NOT_APPLICABLE_DESCRIPTION = "No applicable reason for this classification."


def _extend_enum_with_not_applicable(
    enum_cls: type[StrEnum],
) -> type[StrEnum]:
    """
    Return a copy of `enum_cls` with a `NOT_APPLICABLE` member
    appended. If the member already exists, return the original.
    """
    if hasattr(enum_cls, "NOT_APPLICABLE"):
        return enum_cls
    members = [(m.name, m.value) for m in enum_cls]
    members.append(("NOT_APPLICABLE", NOT_APPLICABLE_DESCRIPTION))
    return StrEnum(enum_cls.__name__, members)


class Property(BaseModel, Generic[T]):
    """
    A generic base class for properties used by Detector agents.

    Subclasses should specify the type parameter T for type-safe value
    extraction. The reason field must be a StrEnum defining the possible
    classification reasons.

    By default, a NOT_APPLICABLE member is automatically added to the
    reason enum. Set ``NOT_APPLICABLE = False`` on the subclass to
    disable this behavior.

    Example:
        class TemperatureReason(StrEnum):
            TOO_COLD = "The temperature is too cold."
            TOO_HOT = "The temperature is too hot."
            COMFORTABLE = "The temperature is comfortable."
            # NOT_APPLICABLE is added automatically

        class Temperature(Property[float]):
            reason: TemperatureReason

        # Opt out:
        class StrictTemp(Property[float]):
            NOT_APPLICABLE = False
            reason: TemperatureReason
    """

    NOT_APPLICABLE: ClassVar[bool] = True

    value: T | None = Field(
        default=None, description="The extracted value of the property."
    )
    reason: StrEnum = Field(
        description="The categorical reason for the classification."
    )

    def __init_subclass__(cls, **kwargs):
        """
        Automatically extend the reason enum with NOT_APPLICABLE
        unless ``NOT_APPLICABLE = False`` is set on the subclass.

        This hook runs before Pydantic processes the class, allowing
        us to modify the reason annotation to include the fallback.
        """
        super().__init_subclass__(**kwargs)

        if not cls.__dict__.get("NOT_APPLICABLE", True):
            return

        if "reason" not in cls.__annotations__:
            return

        reason_enum = cls.__annotations__["reason"]
        if isinstance(reason_enum, type) and issubclass(reason_enum, StrEnum):
            cls.__annotations__["reason"] = _extend_enum_with_not_applicable(
                reason_enum
            )


class ClassificationProperty(BaseModel):
    """
    A base class for classification outputs used by Classifier agents.

    Subclasses should define category and reason StrEnums, plus a
    CATEGORY_REASONS mapping that links valid reasons to each category.

    By default, a NOT_APPLICABLE member is automatically added to both
    category and reason enums. Set ``NOT_APPLICABLE = False`` on the
    subclass to disable this behavior.

    Example:
        class SentimentCategory(StrEnum):
            POSITIVE = "Positive sentiment."
            NEGATIVE = "Negative sentiment."

        class SentimentReason(StrEnum):
            ENTHUSIASTIC = "Expresses enthusiasm."
            FRUSTRATED = "Expresses frustration."

        class Sentiment(ClassificationProperty):
            category: SentimentCategory
            reason: SentimentReason

            CATEGORY_REASONS = {
                SentimentCategory.POSITIVE: [SentimentReason.ENTHUSIASTIC],
                SentimentCategory.NEGATIVE: [SentimentReason.FRUSTRATED],
            }

        # Opt out:
        class StrictSentiment(ClassificationProperty):
            NOT_APPLICABLE = False
            category: SentimentCategory
            reason: SentimentReason
    """

    NOT_APPLICABLE: ClassVar[bool] = True

    category: StrEnum = Field(description="The classification category.")
    reason: StrEnum = Field(
        description="The justification for the classification."
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Classification confidence (0.0–1.0).",
    )

    # Subclasses define mapping of valid reasons per category
    CATEGORY_REASONS: ClassVar[dict[StrEnum, list[StrEnum]]] = {}

    def __init_subclass__(cls, **kwargs):
        """
        Automatically extend category and reason enums with
        NOT_APPLICABLE unless ``NOT_APPLICABLE = False`` is set.

        This hook runs before Pydantic processes the class, allowing
        us to modify the annotations to include the fallback members.
        """
        super().__init_subclass__(**kwargs)

        if not cls.__dict__.get("NOT_APPLICABLE", True):
            return

        for field in ("category", "reason"):
            if field not in cls.__annotations__:
                continue
            enum_cls = cls.__annotations__[field]
            if isinstance(enum_cls, type) and issubclass(enum_cls, StrEnum):
                cls.__annotations__[field] = _extend_enum_with_not_applicable(
                    enum_cls
                )

    @model_validator(mode="after")
    def validate_category_reason_link(self) -> "ClassificationProperty":
        """Ensure reason is valid for the selected category."""
        if not self.CATEGORY_REASONS:
            return self  # No validation if mapping not defined

        # NOT_APPLICABLE category can have any reason
        if (
            hasattr(self.category, "name")
            and self.category.name == "NOT_APPLICABLE"
        ):
            return self

        valid_reasons = self.CATEGORY_REASONS.get(self.category, [])
        # Allow NOT_APPLICABLE reason for any category
        if (
            hasattr(self.reason, "name")
            and self.reason.name == "NOT_APPLICABLE"
        ):
            return self

        if valid_reasons and self.reason not in valid_reasons:
            raise PropertyConfigurationError(
                f"Reason '{self.reason.name}' is not valid for "
                f"category '{self.category.name}'. "
                f"Valid reasons: {[r.name for r in valid_reasons]}"
            )
        return self
