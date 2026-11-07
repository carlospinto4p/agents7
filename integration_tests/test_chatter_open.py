"""End-to-end check that an agent can speak first.

The opening instruction is sent to the provider as a user
message that never enters history, so this test is what proves
a real model answers it with an in-character greeting rather
than repeating the instruction back.
"""

import pytest
from integration_helpers import warm_model

from agents7.agents import Chatter


@pytest.fixture
def opener() -> Chatter:
    """Demo chatter allowed to start the conversation."""
    agent = Chatter.from_yaml("examples/configs/demo_chatter.yml")
    warm_model(agent.engine.model_name)
    return agent.with_overrides(can_open=True)


def test_open_chat_produces_a_first_turn(opener: Chatter) -> None:
    """The agent writes an opening line with nothing to answer."""
    response = opener.open_chat()

    assert response.response.strip(), "Expected a non-empty opening line"
    assert "[Write the first message" not in response.response, (
        "The model echoed the opening instruction instead of answering it"
    )


def test_open_chat_history_holds_only_the_agents_line(
    opener: Chatter,
) -> None:
    """The instruction is not stored — the transcript starts clean."""
    opener.open_chat()

    assert len(opener.history) == 1, (
        f"Expected one message, got {opener.history}"
    )
    assert opener.history[0]["role"] == "assistant"
