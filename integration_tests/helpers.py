"""Shared helpers for integration tests."""


def assert_not_applicable(agent, inputs: list[str]) -> None:
    """Assert that the agent classifies all inputs as
    NOT_APPLICABLE.

    Runs `agent.act()` on each input and checks that the
    resulting model has `reason.name == "NOT_APPLICABLE"`.

    :param agent: A Detector, Classifier, or Regressor agent.
    :param inputs: List of input texts that should be
        irrelevant to the agent's domain.
    """
    for text in inputs:
        result = agent.act({"text": text})
        assert result.model is not None, f"Model is None for input: {text}"
        assert result.model.reason.name == "NOT_APPLICABLE", (
            f"Expected NOT_APPLICABLE for '{text}', "
            f"got {result.model.reason.name}"
        )
