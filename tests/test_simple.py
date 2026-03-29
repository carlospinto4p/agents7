"""Unit tests for agents7.simple convenience functions."""

import json


from agents7.agents.classifier import Classifier
from agents7.models.base import ClassificationProperty
from agents7.simple import simple_classifier
from agents7.testing import create_mock_engine


# --- simple_classifier ---


def test_creates_classifier():
    """Returns a Classifier instance."""
    engine = create_mock_engine()
    agent = simple_classifier(
        categories=["POS", "NEG"],
        system_prompt="Classify.",
        engine=engine,
    )
    assert isinstance(agent, Classifier)
    assert agent.name == "classifier"
    assert agent.version == 1


def test_custom_name_and_version():
    """Accepts custom name and version."""
    engine = create_mock_engine()
    agent = simple_classifier(
        categories=["A", "B"],
        system_prompt="Test.",
        engine=engine,
        name="my-agent",
        version=3,
    )
    assert agent.name == "my-agent"
    assert agent.version == 3


def test_output_model_is_classification_property():
    """Generated output model is a ClassificationProperty."""
    engine = create_mock_engine()
    agent = simple_classifier(
        categories=["X", "Y"],
        system_prompt="Test.",
        engine=engine,
    )
    assert issubclass(agent.output_model, ClassificationProperty)


def test_categories_in_enum():
    """All categories are in the output model's enum."""
    engine = create_mock_engine()
    agent = simple_classifier(
        categories=["SPAM", "HAM"],
        system_prompt="Test.",
        engine=engine,
    )
    cats = agent.possible_categories
    assert "SPAM" in cats
    assert "HAM" in cats


def test_not_applicable_added_by_default():
    """NOT_APPLICABLE is added when not_applicable=True."""
    engine = create_mock_engine()
    agent = simple_classifier(
        categories=["A", "B"],
        system_prompt="Test.",
        engine=engine,
    )
    cats = agent.possible_categories
    assert "NOT_APPLICABLE" in cats


def test_not_applicable_disabled():
    """NOT_APPLICABLE omitted when not_applicable=False."""
    engine = create_mock_engine()
    agent = simple_classifier(
        categories=["A", "B"],
        system_prompt="Test.",
        engine=engine,
        not_applicable=False,
    )
    cats = agent.possible_categories
    assert "NOT_APPLICABLE" not in cats


def test_default_reasons_generated():
    """Default reasons are IS_<category>."""
    engine = create_mock_engine()
    agent = simple_classifier(
        categories=["GOOD", "BAD"],
        system_prompt="Test.",
        engine=engine,
    )
    reasons = agent.possible_reasons
    assert "IS_GOOD" in reasons
    assert "IS_BAD" in reasons


def test_custom_reasons():
    """Custom reasons override defaults."""
    engine = create_mock_engine()
    agent = simple_classifier(
        categories=["A", "B"],
        reasons=["REASON_X", "REASON_Y"],
        system_prompt="Test.",
        engine=engine,
    )
    reasons = agent.possible_reasons
    assert "REASON_X" in reasons
    assert "REASON_Y" in reasons


def test_classify_with_mock():
    """Agent classifies using the mock engine."""
    response_json = json.dumps(
        {
            "category": "SPAM",
            "reason": "IS_SPAM",
        }
    )
    engine = create_mock_engine(response_json)
    agent = simple_classifier(
        categories=["SPAM", "HAM"],
        system_prompt="Is this spam?",
        engine=engine,
    )
    result = agent.act({"text": "Buy now!!!"})
    assert result.model is not None
    assert result.model.category.name == "SPAM"
    assert result.model.reason.name == "IS_SPAM"


def test_think_parameter():
    """think parameter is passed through."""
    engine = create_mock_engine()
    agent = simple_classifier(
        categories=["A"],
        system_prompt="Test.",
        engine=engine,
        think=True,
    )
    assert agent.think is True


def test_extra_kwargs_passed():
    """Extra kwargs like prompt_params are forwarded."""
    engine = create_mock_engine()
    agent = simple_classifier(
        categories=["A"],
        system_prompt="Hello $name.",
        engine=engine,
        prompt_params={"name": "World"},
    )
    assert "World" in agent.resolved_system_prompt
