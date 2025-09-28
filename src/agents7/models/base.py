from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class LLMMetrics(BaseModel):
    """
    LLM metrics class.
    """

    tokens: int = Field(description="The number of tokens in the response.")
    prompt_tokens: int = Field(
        description="The number of tokens in the prompt."
    )
    total_duration: float = Field(
        description="The total duration of the response, in seconds."
    )


class AgentResponse(BaseModel):
    """
    Agent response class.
    """

    response: str = Field(description="The generated text response.")
    model: BaseModel = Field(
        description="The structured output validated by the output model."
    )
    thinking: str | None = Field(
        default=None, description="The thinking part of the response, if any."
    )
    metrics: LLMMetrics = Field(description="Metrics of the LLM request.")
    idx: int | None = Field(
        default=None, description="Index for tracking purposes."
    )


class Property(BaseModel):
    """
    A base class for properties that can be used in agents.
    This class has to be extended to create specific properties.
    """

    value: Any = Field(description="The value of the property.")
    reason: StrEnum = Field(
        description="Reasons for the value of the property."
    )
