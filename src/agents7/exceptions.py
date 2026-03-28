class Agents7Exception(Exception):
    """Base exception for Agents7."""


class LLMException(Agents7Exception):
    """Base exception for LLM-related errors."""


class ParseError(LLMException):
    """Raised when LLM response parsing fails."""


class ProviderError(LLMException):
    """Raised when the LLM provider returns an error."""


class ModelUnavailableError(ProviderError):
    """Raised when the requested model is not found."""


class RateLimitError(ProviderError):
    """Raised when the provider returns a rate-limit error."""


class PropertyConfigurationError(Agents7Exception):
    """Exception raised when a Property subclass is misconfigured."""


class PromptParameterError(Agents7Exception):
    """Exception raised when prompt_params values are invalid."""


class AgentConfigurationError(Agents7Exception):
    """Exception raised when an agent is misconfigured."""


class PipelineConfigurationError(Agents7Exception):
    """Exception raised when a Pipeline is misconfigured."""


class SuggestionError(Agents7Exception):
    """Raised for invalid suggestion operations."""


class ChatEndedError(Agents7Exception):
    """Raised when a Chatter conversation has ended."""
