from enum import Enum
from typing import Any, get_origin

from pydantic import BaseModel, Field


class PropertyReason(Enum):
    """
    Enum for the reasons of a property value.
    This class has to be extended to create specific reasons.
    """

    NOT_APPLICABLE = "no_applicable"


class Property(BaseModel):
    """
    A base class for properties that can be used in agents.
    This class has to be extended to create specific properties.
    """

    name: str = Field(
        ...,
        description="The name of the property. It should be unique "
        "across all properties.",
    )
    value: Any = Field(..., description="The value of the property.")
    reason: PropertyReason = Field(
        ..., description="The reason for the value of the property. "
    )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Skip base class
        if cls is Property:
            return

        required_fields = {"name", "value", "reason"}
        declared_fields = set(cls.__annotations__)

        missing = required_fields - declared_fields
        if missing:
            raise TypeError(
                f"`{cls.__name__}` must re-declare the fields: "
                f"`{'`, `'.join(missing)}`"
            )

        # Check that 'reason' is annotated with a subclass of PropertyReason
        reason_type = cls.__annotations__.get("reason")
        origin = get_origin(reason_type) or reason_type

        if not isinstance(origin, type) or not issubclass(
            origin, PropertyReason
        ):
            raise TypeError(
                f"The 'reason' field in `{cls.__name__}` must be a subclass "
                f"of `PropertyReason`, got: `{reason_type}`"
            )
