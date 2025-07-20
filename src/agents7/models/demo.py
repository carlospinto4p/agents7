from pydantic import BaseModel


class DemoModel(BaseModel):
    """
    A simple demo model for testing purposes.
    This model can be used to demonstrate the functionality of an agent
    without requiring a complex setup.
    """

    name: str
    description: str
    value: int
