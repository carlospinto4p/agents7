import pytest

from agents7.agents.classifier import Classifier
from agents7.agents.detector import Detector
from agents7.agents.regressor import Regressor


# --- Test data (3-item subsets) ---

detector_data = [
    {
        "text": "This is a test text.",
        "expected_reason": "NOT_APPLICABLE",
    },
    {
        "text": "The water in the lake is starting to freeze.",
        "expected_reason": "TOO_COLD",
    },
    {
        "text": ("It's scorching outside; I can barely stay in the sun."),
        "expected_reason": "TOO_HOT",
    },
]

classifier_data = [
    {
        "text": "I absolutely love this! Best day ever!",
        "expected_category": "POSITIVE",
    },
    {
        "text": "This is terrible and I'm very disappointed.",
        "expected_category": "NEGATIVE",
    },
    {
        "text": "The meeting is at 3 PM in conference room B.",
        "expected_category": "NEUTRAL",
    },
]

regressor_data = [
    {
        "text": "This is a test text.",
        "expected_reason": "NOT_APPLICABLE",
    },
    {
        "text": "The water in the lake is starting to freeze.",
        "expected_reason": "TOO_COLD",
    },
    {
        "text": ("It's scorching outside; I can barely stay in the sun."),
        "expected_reason": "TOO_HOT",
    },
]


# --- Fixtures ---


@pytest.fixture
def temperature_detector():
    """Create a temperature detector instance."""
    return Detector.from_yaml(
        "examples/configs/demo_temperature_detector.yml"
    )


@pytest.fixture
def sentiment_classifier():
    """Create a sentiment classifier instance."""
    return Classifier.from_yaml(
        "examples/configs/demo_sentiment_classifier.yml"
    )


@pytest.fixture
def temperature_regressor():
    """Create a temperature regressor instance."""
    return Regressor.from_yaml(
        "examples/configs/demo_temperature_detector.yml"
    )


# --- Detector async tests ---


class TestDetectorAsync:
    """Async tests for the Detector agent."""

    async def test_act_async(self, temperature_detector: Detector) -> None:
        """Test single async detection returns correct reason."""
        case = detector_data[1]  # freezing lake
        response = await temperature_detector.act_async(
            {"text": case["text"]}
        )
        assert response.model.reason.name == case["expected_reason"]

    async def test_act_in_batch_async(
        self, temperature_detector: Detector
    ) -> None:
        """Test batch async detection returns correct count and reasons."""
        inputs = [{"text": c["text"]} for c in detector_data]
        responses = await temperature_detector.act_in_batch_async(
            inputs, with_tqdm=False
        )

        assert len(responses) == len(detector_data)
        for case, output in zip(detector_data, responses):
            assert output.model.reason.name == case["expected_reason"]

    async def test_act_in_batch_async_empty(
        self, temperature_detector: Detector
    ) -> None:
        """Test that empty input returns empty list."""
        responses = await temperature_detector.act_in_batch_async(
            [], with_tqdm=False
        )
        assert responses == []


# --- Classifier async tests ---


class TestClassifierAsync:
    """Async tests for the Classifier agent."""

    async def test_act_async(self, sentiment_classifier: Classifier) -> None:
        """Test single async classification returns correct category."""
        case = classifier_data[0]  # positive sentiment
        response = await sentiment_classifier.act_async(
            {"text": case["text"]}
        )
        assert response.model.category.name == case["expected_category"]

    async def test_act_in_batch_async(
        self, sentiment_classifier: Classifier
    ) -> None:
        """Test batch async classification returns correct count and categories."""
        inputs = [{"text": c["text"]} for c in classifier_data]
        responses = await sentiment_classifier.act_in_batch_async(
            inputs, with_tqdm=False
        )

        assert len(responses) == len(classifier_data)
        for case, output in zip(classifier_data, responses):
            assert output.model.category.name == case["expected_category"]


# --- Regressor async tests ---


class TestRegressorAsync:
    """Async tests for the Regressor agent."""

    async def test_act_async(self, temperature_regressor: Regressor) -> None:
        """Test single async regression returns correct reason."""
        case = regressor_data[1]  # freezing lake
        response = await temperature_regressor.act_async(
            {"text": case["text"]}
        )
        assert response.model.reason.name == case["expected_reason"]

    async def test_act_in_batch_async(
        self, temperature_regressor: Regressor
    ) -> None:
        """Test batch async regression returns correct count and reasons."""
        inputs = [{"text": c["text"]} for c in regressor_data]
        responses = await temperature_regressor.act_in_batch_async(
            inputs, with_tqdm=False
        )

        assert len(responses) == len(regressor_data)
        for case, output in zip(regressor_data, responses):
            assert output.model.reason.name == case["expected_reason"]
