"""Token pricing for cost estimation.

A provider bills prompt tokens and completion tokens at
different rates, and most now bill a *repeated* prompt far more
cheaply than a new one. `Pricing` carries those three rates so
`cost_cents` can be estimated per call.

Rates are **US dollars per million tokens**, which is how every
provider publishes them — copy the published number straight in
rather than converting it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pricing:
    """Per-million-token rates for one model, in USD.

    :param input: Rate for prompt tokens the provider had to
        read fresh (a cache miss, or a provider with no prompt
        cache at all).
    :param output: Rate for the tokens the model generated.
    :param cached_input: Rate for prompt tokens the provider
        served from its own prompt cache. Defaults to `input`,
        which is the correct value for a provider that does not
        discount repeated prompts. Setting it matters a lot for
        an agent with a long fixed system prompt: the whole
        persona is resent on every turn, so after the first turn
        almost every prompt token is a cache hit.

    The rates a config declares are the ones charged. A provider
    that varies its rates by time of day is not modelled here —
    declare the rate you want the estimate against and read
    `cost_cents` as an estimate, which is what it is called.
    """

    input: float
    output: float
    cached_input: float | None = None

    def __post_init__(self) -> None:
        if self.cached_input is None:
            object.__setattr__(self, "cached_input", self.input)
        for name in ("input", "output", "cached_input"):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"Pricing.{name} must not be negative, "
                    f"got {getattr(self, name)}"
                )

    @classmethod
    def flat(cls, rate: float) -> "Pricing":
        """Build a `Pricing` charging one rate for everything.

        :param rate: USD per million tokens, applied to prompt
            and completion alike.
        """
        return cls(input=rate, output=rate, cached_input=rate)

    def cost_cents(
        self,
        prompt_tokens: int,
        tokens: int,
        cached_prompt_tokens: int = 0,
    ) -> float:
        """Estimate one call's cost, in US cents.

        :param prompt_tokens: Total prompt tokens, cached ones
            included — that is what every provider reports.
        :param tokens: Completion tokens.
        :param cached_prompt_tokens: How many of *prompt_tokens*
            were served from the provider's prompt cache.
        :return: Estimated cost in cents.
        """
        cached = min(max(cached_prompt_tokens, 0), max(prompt_tokens, 0))
        fresh = max(prompt_tokens, 0) - cached
        usd = (
            fresh * self.input
            + cached * self.cached_input
            + max(tokens, 0) * self.output
        ) / 1_000_000
        return usd * 100
