from enum import StrEnum

import pytest

from agents7.agents.classifier import Classifier
from agents7.agents.detector import Detector
from agents7.agents.labeler import Labeler
from agents7.agents.regressor import Regressor
from agents7.exceptions import PromptParameterError
from agents7.models import ClassificationProperty, Property

from helpers import DummyReason, make_agent, make_labeler


class DummyProperty(Property[float]):
    reason: DummyReason


class DummyCategory(StrEnum):
    CAT_A = "Category A."
    CAT_B = "Category B."


class DummyClassReason(StrEnum):
    REASON_A = "Reason A."
    REASON_B = "Reason B."


class DummyClassification(ClassificationProperty):
    category: DummyCategory
    reason: DummyClassReason

    CATEGORY_REASONS = {
        DummyCategory.CAT_A: [DummyClassReason.REASON_A],
        DummyCategory.CAT_B: [DummyClassReason.REASON_B],
    }


# --- Backward compatibility ---


def test_no_params_unchanged_behavior():
    """Labeler without prompt_params behaves identically."""
    agent = make_labeler(system_prompt="Analyze this data.")
    assert agent.resolved_system_prompt == "Analyze this data."
    assert "Analyze this data." in agent.prompt


def test_none_params_unchanged():
    """Explicit None for prompt_params behaves identically."""
    agent = make_labeler(
        system_prompt="Analyze this data.",
        prompt_params=None,
    )
    assert agent.resolved_system_prompt == "Analyze this data."


# --- Static param substitution ---


def test_static_param_substitution():
    """Static string params are substituted in the prompt."""
    agent = make_labeler(
        system_prompt="Analyze data for $company on $date.",
        prompt_params={"company": "Acme Corp", "date": "2026-02-01"},
    )
    resolved = agent.resolved_system_prompt
    assert "Acme Corp" in resolved
    assert "2026-02-01" in resolved
    assert "$company" not in resolved
    assert "$date" not in resolved


def test_static_params_in_full_prompt():
    """Static params are resolved in the full prompt property."""
    agent = make_labeler(
        system_prompt="Analyze data for $company.",
        prompt_params={"company": "TestCo"},
    )
    assert "TestCo" in agent.prompt
    assert "$company" not in agent.prompt


# --- Callable param evaluation ---


def test_callable_param_evaluation():
    """Callable values are called each time the prompt is resolved."""
    call_count = 0

    def counter():
        nonlocal call_count
        call_count += 1
        return f"call-{call_count}"

    agent = make_labeler(
        system_prompt="Value is $val.",
        prompt_params={"val": counter},
    )
    first = agent.resolved_system_prompt
    second = agent.resolved_system_prompt
    assert "call-1" in first
    assert "call-2" in second
    assert call_count == 2


def test_mixed_static_and_callable_params():
    """Mix of static strings and callables works correctly."""
    agent = make_labeler(
        system_prompt="$greeting from $source.",
        prompt_params={
            "greeting": "Hello",
            "source": lambda: "dynamic",
        },
    )
    resolved = agent.resolved_system_prompt
    assert "Hello" in resolved
    assert "dynamic" in resolved


# --- Unmatched placeholders ---


def test_unmatched_placeholders_preserved():
    """Unmatched $placeholders are left as-is via safe_substitute."""
    agent = make_labeler(
        system_prompt="Hello $name, welcome to $place.",
        prompt_params={"name": "Alice"},
    )
    resolved = agent.resolved_system_prompt
    assert "Alice" in resolved
    assert "$place" in resolved


# --- Empty params dict ---


def test_empty_params_dict():
    """Empty dict for prompt_params works same as None."""
    agent = make_labeler(
        system_prompt="No changes here.",
        prompt_params={},
    )
    assert agent.resolved_system_prompt == "No changes here."


# --- Batch prompts ---


def test_params_applied_to_prompt_for_batch():
    """Params are resolved in prompt_for_batch as well."""
    agent = make_labeler(
        system_prompt="Batch for $company.",
        prompt_params={"company": "BatchCo"},
    )
    assert "BatchCo" in agent.prompt_for_batch
    assert "$company" not in agent.prompt_for_batch


# --- All agent types ---


def test_detector_param_substitution():
    """Detector resolves prompt_params in prompt."""
    agent = make_agent(
        Detector,
        output_model=DummyProperty,
        system_prompt="Detect for $target.",
        prompt_params={"target": "anomalies"},
    )
    assert "anomalies" in agent.prompt
    assert "$target" not in agent.prompt
    assert "anomalies" in agent.prompt_for_batch
    assert "$target" not in agent.prompt_for_batch


def test_classifier_param_substitution():
    """Classifier resolves prompt_params in prompt."""
    agent = make_agent(
        Classifier,
        output_model=DummyClassification,
        system_prompt="Classify $domain data.",
        prompt_params={"domain": "finance"},
    )
    assert "finance" in agent.prompt
    assert "$domain" not in agent.prompt
    assert "finance" in agent.prompt_for_batch
    assert "$domain" not in agent.prompt_for_batch


def test_regressor_param_substitution():
    """Regressor resolves prompt_params in prompt."""
    agent = make_agent(
        Regressor,
        output_model=DummyProperty,
        system_prompt="Extract $metric values.",
        prompt_params={"metric": "temperature"},
    )
    assert "temperature" in agent.prompt
    assert "$metric" not in agent.prompt
    assert "temperature" in agent.prompt_for_batch
    assert "$metric" not in agent.prompt_for_batch


# --- Validation errors ---


def test_invalid_param_value_type():
    """Non-string, non-callable values raise PromptParameterError."""
    with pytest.raises(PromptParameterError):
        make_labeler(
            system_prompt="Test $val.",
            prompt_params={"val": 123},
        )


def test_invalid_param_value_list():
    """List values raise PromptParameterError."""
    with pytest.raises(PromptParameterError):
        make_labeler(
            system_prompt="Test $val.",
            prompt_params={"val": ["a", "b"]},
        )


def test_invalid_params_not_dict():
    """Non-dict prompt_params raises PromptParameterError."""
    with pytest.raises(PromptParameterError):
        make_labeler(
            system_prompt="Test.",
            prompt_params="not a dict",
        )


# --- YAML loading ---


def test_yaml_loading_with_prompt_params(tmp_path):
    """YAML config with prompt_params is parsed correctly."""
    yaml_content = (
        "name: TestAgent\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: |\n"
        "  Analyze data for $company on $date.\n"
        "prompt_params:\n"
        "    company: Acme Corp\n"
        "    date: 2026-02-01\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )
    yaml_file = tmp_path / "test_config.yml"
    yaml_file.write_text(yaml_content)

    agent = Labeler.from_yaml(str(yaml_file))
    assert agent.prompt_params == {
        "company": "Acme Corp",
        "date": "2026-02-01",
    }
    assert "Acme Corp" in agent.resolved_system_prompt
    assert "2026-02-01" in agent.resolved_system_prompt


def test_yaml_loading_without_prompt_params(tmp_path):
    """YAML config without prompt_params sets it to None."""
    yaml_content = (
        "name: TestAgent\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: |\n"
        "  Analyze data.\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )
    yaml_file = tmp_path / "test_config.yml"
    yaml_file.write_text(yaml_content)

    agent = Labeler.from_yaml(str(yaml_file))
    assert agent.prompt_params is None


def test_yaml_values_cast_to_str(tmp_path):
    """YAML numeric values are cast to strings."""
    yaml_content = (
        "name: TestAgent\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: |\n"
        "  Threshold is $threshold.\n"
        "prompt_params:\n"
        "    threshold: 42\n"
        "output_model: agents7.models.demo.PersonInfo\n"
    )
    yaml_file = tmp_path / "test_config.yml"
    yaml_file.write_text(yaml_content)

    agent = Labeler.from_yaml(str(yaml_file))
    assert agent.prompt_params["threshold"] == "42"
    assert isinstance(agent.prompt_params["threshold"], str)
    assert "42" in agent.resolved_system_prompt


def test_detector_yaml_loading_with_prompt_params(tmp_path):
    """Detector YAML config with prompt_params is parsed correctly."""
    yaml_content = (
        "name: TestDetector\n"
        "version: 1\n"
        "engine:\n"
        "    provider: ollama\n"
        "    model: qwen3\n"
        "system_prompt: |\n"
        "  Detect $target properties.\n"
        "prompt_params:\n"
        "    target: temperature\n"
        "output_model: agents7.models.demo.Temperature\n"
    )
    yaml_file = tmp_path / "test_config.yml"
    yaml_file.write_text(yaml_content)

    agent = Detector.from_yaml(str(yaml_file))
    assert agent.prompt_params == {"target": "temperature"}
    assert "temperature" in agent.resolved_system_prompt
