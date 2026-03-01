from agents7.agents import Classifier, Detector, Regressor

from data import SENTIMENT_DATA, TEMPERATURE_DATA

# Async tests use 3-item subsets (no COMFORTABLE case)
_detector_data = TEMPERATURE_DATA[:3]
_regressor_data = TEMPERATURE_DATA[:3]


# --- Detector async tests ---


class TestDetectorAsync:
    """Async tests for the Detector agent."""

    async def test_act_async(self, temperature_detector: Detector) -> None:
        """Test single async detection returns correct reason."""
        case = _detector_data[1]  # freezing lake
        response = await temperature_detector.act_async(
            {"text": case["text"]}
        )
        assert response.model.reason.name == case["expected_reason"]

    async def test_act_in_batch_async(
        self, temperature_detector: Detector
    ) -> None:
        """Test batch async detection returns correct reasons."""
        inputs = [{"text": c["text"]} for c in _detector_data]
        responses = await temperature_detector.act_in_batch_async(
            inputs, with_tqdm=False
        )

        assert len(responses) == len(_detector_data)
        for case, output in zip(_detector_data, responses):
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
        """Test single async classification."""
        case = SENTIMENT_DATA[0]  # positive sentiment
        response = await sentiment_classifier.act_async(
            {"text": case["text"]}
        )
        assert response.model.category.name == case["expected_category"]

    async def test_act_in_batch_async(
        self, sentiment_classifier: Classifier
    ) -> None:
        """Test batch async classification."""
        inputs = [{"text": c["text"]} for c in SENTIMENT_DATA]
        responses = await sentiment_classifier.act_in_batch_async(
            inputs, with_tqdm=False
        )

        assert len(responses) == len(SENTIMENT_DATA)
        for case, output in zip(SENTIMENT_DATA, responses):
            assert output.model.category.name == case["expected_category"]


# --- Regressor async tests ---


class TestRegressorAsync:
    """Async tests for the Regressor agent."""

    async def test_act_async(self, temperature_regressor: Regressor) -> None:
        """Test single async regression returns correct reason."""
        case = _regressor_data[1]  # freezing lake
        response = await temperature_regressor.act_async(
            {"text": case["text"]}
        )
        assert response.model.reason.name == case["expected_reason"]

    async def test_act_in_batch_async(
        self, temperature_regressor: Regressor
    ) -> None:
        """Test batch async regression returns correct reasons."""
        inputs = [{"text": c["text"]} for c in _regressor_data]
        responses = await temperature_regressor.act_in_batch_async(
            inputs, with_tqdm=False
        )

        assert len(responses) == len(_regressor_data)
        for case, output in zip(_regressor_data, responses):
            assert output.model.reason.name == case["expected_reason"]
