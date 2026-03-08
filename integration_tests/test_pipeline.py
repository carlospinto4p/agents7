import pytest

from agents7.agents.pipeline import Pipeline, PipelineResponse
from agents7.models.demo import PersonBio, PersonInfo

from data import PIPELINE_DATA


@pytest.fixture
def person_bio_pipeline():
    """Fixture to create a person-to-bio pipeline."""
    return Pipeline.from_yaml("examples/configs/demo_pipeline.yml")


# --- Structure ---


def test_pipeline_from_yaml_structure(
    person_bio_pipeline: Pipeline,
) -> None:
    """Test that the pipeline loads with correct structure."""
    assert len(person_bio_pipeline.steps) == 2, (
        f"Expected 2 steps, got {len(person_bio_pipeline.steps)}"
    )
    assert person_bio_pipeline.steps[0].agent.name == ("PersonExtractor")
    assert person_bio_pipeline.steps[1].agent.name == ("BioGenerator")


# --- Single execution ---


@pytest.mark.parametrize("test_case", PIPELINE_DATA)
def test_single_pipeline_execution(
    person_bio_pipeline: Pipeline, test_case: dict
) -> None:
    """Test single pipeline execution end-to-end."""
    input_data = {"text": test_case["text"]}
    response = person_bio_pipeline.act(input_data)

    assert isinstance(response, PipelineResponse)
    assert response.success, f"Pipeline failed for: '{test_case['text']}'"
    assert len(response.steps) == 2, (
        f"Expected 2 steps, got {len(response.steps)}"
    )
    assert isinstance(response.model, PersonBio), (
        f"Expected PersonBio, got {type(response.model).__name__}"
    )
    assert test_case["expected_name"].lower() in (
        response.model.bio.lower()
    ), (
        f"Expected '{test_case['expected_name']}' in bio, "
        f"got: '{response.model.bio}'"
    )


def test_pipeline_intermediate_step(
    person_bio_pipeline: Pipeline,
) -> None:
    """Test that the intermediate step produces PersonInfo."""
    input_data = {"text": PIPELINE_DATA[0]["text"]}
    response = person_bio_pipeline.act(input_data)

    assert response.success, "Pipeline should succeed"
    assert len(response.steps) >= 1, "Should have at least 1 step"

    step_0_model = response.steps[0].model
    assert isinstance(step_0_model, PersonInfo), (
        f"Step 0 should produce PersonInfo, got {type(step_0_model).__name__}"
    )
    assert step_0_model.name is not None, (
        "Step 0 PersonInfo should have a non-null name"
    )


# --- Batch processing ---


def test_pipeline_batch(
    person_bio_pipeline: Pipeline,
) -> None:
    """Test batch pipeline execution."""
    input_list = [{"text": case["text"]} for case in PIPELINE_DATA]
    responses = person_bio_pipeline.act_in_batch(input_list)

    assert len(responses) == len(PIPELINE_DATA), (
        f"Expected {len(PIPELINE_DATA)} responses but got {len(responses)}"
    )

    for i, (test_case, response) in enumerate(zip(PIPELINE_DATA, responses)):
        assert isinstance(response, PipelineResponse)
        assert response.success, (
            f"Pipeline failed for case {i}: '{test_case['text']}'"
        )
        assert isinstance(response.model, PersonBio), (
            f"Expected PersonBio for case {i}, "
            f"got {type(response.model).__name__}"
        )
        assert test_case["expected_name"].lower() in (
            response.model.bio.lower()
        ), (
            f"Expected '{test_case['expected_name']}' in bio "
            f"for case {i}, got: '{response.model.bio}'"
        )
