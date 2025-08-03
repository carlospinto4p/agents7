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
        # Ensure that subclasses of Property override the required fields.
        super().__init_subclass__(**kwargs)

        # Skip check for the base class itself
        if cls is Property:
            return

        required_fields = {"name", "value", "reason"}
        overridden_fields = {
            name
            for name, value in cls.__dict__.items()
            if isinstance(value, (property, Field))
            or name in cls.__annotations__
        }
        missing = required_fields - overridden_fields
        if missing:
            raise TypeError(
                f"Subclasses of Property must override the following fields: "
                f"`{'`, `'.join(missing)}`"
            )

        # Enforce that `reason` annotation is a subclass of PropertyReason
        reason_type = cls.__annotations__.get("reason")
        if reason_type is None:
            raise TypeError("Subclasses must define a 'reason' annotation.")

        # Handle direct subclass or Annotated/etc
        origin = get_origin(reason_type) or reason_type
        if not isinstance(origin, type) or not issubclass(
            origin, PropertyReason
        ):
            raise TypeError(
                f"The 'reason' field must be annotated with a subclass of "
                f"`PropertyReason`, got: `{reason_type}`"
            )
