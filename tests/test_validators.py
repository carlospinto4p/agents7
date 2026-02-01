"""Unit tests for validation hooks (no LLM required)."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from agents7.agents.labeler import Labeler
from agents7.llm import LLMEngine


# --- Helper models & fixtures ---


class DummyOutput(BaseModel):
    value: str


ENGINE = LLMEngine(provider="ollama", model_name="qwen3")

DUMMY_METRICS = {
    "tokens": 10,
    "prompt_tokens": 5,
    "total_duration": 0.1,
}


def _make_labeler(**kwargs) -> Labeler:
    defaults = {
        "name": "test",
        "version": 1,
        "engine": ENGINE,
        "system_prompt": "Test prompt.",
        "output_model": DummyOutput,
    }
    defaults.update(kwargs)
    return Labeler(**defaults)


def _upper_validator(model: BaseModel) -> BaseModel:
    """Transforms value to uppercase."""
    return DummyOutput(value=model.value.upper())


def _reject_validator(model: BaseModel) -> BaseModel:
    """Always rejects by raising."""
    raise ValueError("rejected")


def _append_validator(model: BaseModel) -> BaseModel:
    """Appends '!' to value."""
    return DummyOutput(value=model.value + "!")


# ===================================================================
# 1. _run_validators
# ===================================================================


class TestRunValidators:
    def test_no_validators_passthrough(self):
        """Model passes through unchanged when no validators."""
        agent = _make_labeler(validators=None)
        model = DummyOutput(value="hello")
        assert agent._run_validators(model) is model

    def test_none_model_passthrough(self):
        """None model passes through unchanged."""
        agent = _make_labeler(validators=[_upper_validator])
        assert agent._run_validators(None) is None

    def test_single_validator_transforms(self):
        """Single validator transforms the model."""
        agent = _make_labeler(validators=[_upper_validator])
        model = DummyOutput(value="hello")
        result = agent._run_validators(model)
        assert result.value == "HELLO"

    def test_validator_rejects_returns_none(self):
        """Validator that raises returns None."""
        agent = _make_labeler(validators=[_reject_validator])
        model = DummyOutput(value="hello")
        assert agent._run_validators(model) is None

    def test_pipeline_order(self):
        """Validators run in order; output feeds into next."""
        agent = _make_labeler(
            validators=[_upper_validator, _append_validator]
        )
        model = DummyOutput(value="hello")
        result = agent._run_validators(model)
        assert result.value == "HELLO!"

    def test_pipeline_short_circuits_on_failure(self):
        """First failure short-circuits; later validators skipped."""
        agent = _make_labeler(
            validators=[_reject_validator, _append_validator]
        )
        model = DummyOutput(value="hello")
        assert agent._run_validators(model) is None


# ===================================================================
# 2. act() / act_async() with validators
# ===================================================================


def _mock_chat_response(value: str = "hello") -> dict:
    return {
        "role": "assistant",
        "response": json.dumps({"value": value}),
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }


class TestActWithValidators:
    @patch.object(LLMEngine, "chat")
    def test_act_applies_validator(self, mock_chat):
        """act() applies validators to the parsed model."""
        mock_chat.return_value = _mock_chat_response()
        agent = _make_labeler(validators=[_upper_validator])
        result = agent.act({"text": "test"})
        assert result.model.value == "HELLO"

    @patch.object(LLMEngine, "chat")
    def test_act_failing_validator_returns_none_model(self, mock_chat):
        """act() returns None model when validator rejects."""
        mock_chat.return_value = _mock_chat_response()
        agent = _make_labeler(validators=[_reject_validator])
        result = agent.act({"text": "test"})
        assert result.model is None

    @patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_act_async_applies_validator(self, mock_chat):
        """act_async() applies validators to the parsed model."""
        mock_chat.return_value = _mock_chat_response()
        agent = _make_labeler(validators=[_upper_validator])
        result = await agent.act_async({"text": "test"})
        assert result.model.value == "HELLO"


# ===================================================================
# 3. Batch with validators
# ===================================================================


def _mock_batch_response(values: list[str]) -> dict:
    items = [{"value": v} for v in values]
    return {
        "role": "assistant",
        "response": json.dumps({"items": items}),
        "thinking": None,
        "metrics": DUMMY_METRICS,
    }


class TestBatchWithValidators:
    @patch.object(LLMEngine, "chat")
    def test_batch_validates_each_item(self, mock_chat):
        """act_in_batch() validates each individual item."""
        mock_chat.return_value = _mock_batch_response(["hello", "world"])
        agent = _make_labeler(validators=[_upper_validator])
        results = agent.act_in_batch(
            [{"text": "a"}, {"text": "b"}],
            requests_per_prompt=10,
            iter_func=None,
        )
        assert len(results) == 2
        assert results[0].model.value == "HELLO"
        assert results[1].model.value == "WORLD"

    @patch.object(LLMEngine, "chat_async", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_batch_async_validates_each_item(self, mock_chat):
        """act_in_batch_async() validates each individual item."""
        mock_chat.return_value = _mock_batch_response(["hello", "world"])
        agent = _make_labeler(validators=[_upper_validator])
        results = await agent.act_in_batch_async(
            [{"text": "a"}, {"text": "b"}],
            requests_per_prompt=10,
            with_tqdm=False,
        )
        assert len(results) == 2
        assert results[0].model.value == "HELLO"
        assert results[1].model.value == "WORLD"

    @patch.object(LLMEngine, "chat")
    def test_batch_partial_rejection(self, mock_chat):
        """Validator rejects some items, passes others."""

        def _reject_hello(model: BaseModel) -> BaseModel:
            if model.value == "hello":
                raise ValueError("no hello")
            return model

        mock_chat.return_value = _mock_batch_response(["hello", "world"])
        agent = _make_labeler(validators=[_reject_hello])
        results = agent.act_in_batch(
            [{"text": "a"}, {"text": "b"}],
            requests_per_prompt=10,
            iter_func=None,
        )
        assert len(results) == 2
        assert results[0].model is None
        assert results[1].model.value == "world"


# ===================================================================
# 4. YAML parsing
# ===================================================================


class TestValidatorYAML:
    def test_yaml_with_validators(self, tmp_path):
        """YAML validators list resolves dotted paths."""
        yaml_content = (
            "name: ValidatedLabeler\n"
            "version: 1\n"
            "engine:\n"
            "    provider: ollama\n"
            "    model: qwen3\n"
            "system_prompt: Test.\n"
            "output_model: agents7.models.demo.PersonInfo\n"
            "validators:\n"
            "    - tests.test_validators._upper_validator\n"
        )
        yaml_file = tmp_path / "config.yml"
        yaml_file.write_text(yaml_content)

        agent = Labeler.from_yaml(str(yaml_file))
        assert agent.validators is not None
        assert len(agent.validators) == 1
        assert agent.validators[0] is _upper_validator

    def test_yaml_without_validators(self, tmp_path):
        """YAML without validators key produces None."""
        yaml_content = (
            "name: PlainLabeler\n"
            "version: 1\n"
            "engine:\n"
            "    provider: ollama\n"
            "    model: qwen3\n"
            "system_prompt: Test.\n"
            "output_model: agents7.models.demo.PersonInfo\n"
        )
        yaml_file = tmp_path / "config.yml"
        yaml_file.write_text(yaml_content)

        agent = Labeler.from_yaml(str(yaml_file))
        assert agent.validators is None
