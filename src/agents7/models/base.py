from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Property(BaseModel):
    """
    A base class for properties that can be used in agents.
    This class has to be extended to create specific properties.
    """

    value: Any = Field(description="The value of the property.")
    reason: StrEnum = Field(
        description="Reasons for the value of the property."
    )
