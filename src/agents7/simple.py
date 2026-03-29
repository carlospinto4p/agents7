"""Convenience functions for quick agent creation.

Provides shortcuts that auto-generate output models from
plain Python values, reducing boilerplate for the most
common use cases.

Usage::

    from agents7.simple import simple_classifier

    agent = simple_classifier(
        categories=["POSITIVE", "NEGATIVE", "NEUTRAL"],
        system_prompt="Classify the sentiment of the text.",
        engine=engine,
    )
    result = agent.act({"text": "I love this!"})
    print(result.model.category)  # POSITIVE
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from agents7.agents.classifier import Classifier
from agents7.llm.base import LLMEngine
from agents7.models.base import ClassificationProperty


def simple_classifier(
    categories: list[str],
    system_prompt: str,
    engine: LLMEngine,
    *,
    reasons: list[str] | None = None,
    name: str = "classifier",
    version: int = 1,
    not_applicable: bool = True,
    think: bool = False,
    **kwargs,
) -> Classifier:
    """Create a Classifier from a list of category strings.

    Auto-generates the ``StrEnum`` and
    ``ClassificationProperty`` subclasses so you don't
    have to define them manually.

    :param categories: List of category names (e.g.
        ``["POSITIVE", "NEGATIVE", "NEUTRAL"]``).
        Names must be valid Python identifiers in
        UPPER_SNAKE_CASE.
    :param system_prompt: The system prompt describing
        the classification task.
    :param engine: The LLM engine to use.
    :param reasons: Optional list of reason names. If
        omitted, a default set is generated from the
        categories (e.g. ``["IS_POSITIVE",
        "IS_NEGATIVE", "IS_NEUTRAL"]``).
    :param name: Agent name. Defaults to ``"classifier"``.
    :param version: Agent version. Defaults to ``1``.
    :param not_applicable: Whether to auto-add
        NOT_APPLICABLE to enums. Defaults to ``True``.
    :param think: Enable thinking mode. Defaults to
        ``False``.
    :param kwargs: Extra keyword arguments passed to
        the ``Classifier`` constructor (e.g.
        ``examples``, ``prompt_params``).
    :return: A configured ``Classifier`` instance.

    Usage::

        agent = simple_classifier(
            categories=["SPAM", "HAM"],
            system_prompt="Is this email spam or not?",
            engine=engine,
        )
        response = agent.act({"text": "Buy now!!!"})
        print(response.model.category)  # SPAM
    """
    cat_enum = StrEnum(
        "Category",
        {c: c for c in categories},
    )
    if reasons is None:
        reasons = [f"IS_{c}" for c in categories]
    reason_enum = StrEnum(
        "Reason",
        {r: r for r in reasons},
    )

    output_model = type(
        "Output",
        (ClassificationProperty,),
        {
            "__annotations__": {
                "category": cat_enum,
                "reason": reason_enum,
            },
            "category": Field(
                description="The classification category.",
            ),
            "reason": Field(
                description=("The justification for the classification."),
            ),
            "NOT_APPLICABLE": not_applicable,
            "CATEGORY_REASONS": {},
        },
    )

    return Classifier(
        name=name,
        version=version,
        engine=engine,
        system_prompt=system_prompt,
        output_model=output_model,
        think=think,
        **kwargs,
    )
