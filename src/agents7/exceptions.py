class Agents7Exception(Exception):
    """Base exception for Agents7."""


class LLMException(Agents7Exception):
    """Base exception for LLM-related errors."""


class PropertyConfigurationError(Agents7Exception):
    """Exception raised when a Property subclass is misconfigured."""


class PromptParameterError(Agents7Exception):
    """Exception raised when prompt_params values are invalid."""
