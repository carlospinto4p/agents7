import pytest

from agents7.agents import Labeler
from agents7.models.demo import PersonInfo


# --- Test data ---

demo_company_data = [
    {
        "text": "John is a 35-year-old engineer at Acme Corp in Berlin.",
        "expected_name": "John",
    },
    {
        "text": "Sara is an intern at Acme Corp headquarters.",
        "expected_name": "Sara",
    },
    {
        "text": "Maria works at a different company as a designer.",
        "expected_name": None,
    },
]


@pytest.fixture
def company_labeler():
    """Fixture to create a company-scoped person extractor."""
    return Labeler.from_yaml("examples/configs/demo_labeler_params.yml")


# --- Prompt resolution ---


def test_prompt_params_resolved_in_prompt(
    company_labeler: Labeler,
) -> None:
    """Test that prompt_params are substituted in the prompt."""
    prompt = company_labeler.resolved_system_prompt

    assert "Acme Corp" in prompt, "Expected 'Acme Corp' in resolved prompt"
    assert "2026-02-01" in prompt, "Expected '2026-02-01' in resolved prompt"
    assert "$company" not in prompt, (
        "Placeholder '$company' should not remain in resolved prompt"
    )
    assert "$current_date" not in prompt, (
        "Placeholder '$current_date' should not remain in resolved prompt"
    )


# --- Single extraction ---


@pytest.mark.parametrize(
    "test_case",
    [c for c in demo_company_data if c["expected_name"] is not None],
)
def test_extracts_company_person(
    company_labeler: Labeler, test_case: dict
) -> None:
    """Test extraction of people related to Acme Corp."""
    input_data = {"text": test_case["text"]}
    response = company_labeler.act(input_data)

    assert response.model is not None, (
        f"Expected a model for: '{test_case['text']}'"
    )
    assert isinstance(response.model, PersonInfo)
    assert response.model.name is not None, (
        f"Expected non-null name for: '{test_case['text']}'"
    )
    assert test_case["expected_name"].lower() in (
        response.model.name.lower()
    ), (
        f"Expected name '{test_case['expected_name']}' "
        f"but got '{response.model.name}' "
        f"for text: '{test_case['text']}'"
    )


def test_rejects_non_company_person(
    company_labeler: Labeler,
) -> None:
    """Test that non-Acme Corp people return null fields."""
    non_company = next(
        c for c in demo_company_data if c["expected_name"] is None
    )
    input_data = {"text": non_company["text"]}
    response = company_labeler.act(input_data)

    assert response.model is not None, "Expected a model response"
    assert isinstance(response.model, PersonInfo)
    assert response.model.name is None, (
        f"Expected null name for non-company person, "
        f"got '{response.model.name}'"
    )


# --- Batch processing ---


def test_prompt_params_batch(
    company_labeler: Labeler,
) -> None:
    """Test batch processing with prompt_params."""
    input_list = [{"text": case["text"]} for case in demo_company_data]
    responses = company_labeler.act_in_batch(input_list)

    assert len(responses) == len(demo_company_data), (
        f"Expected {len(demo_company_data)} responses "
        f"but got {len(responses)}"
    )

    for i, (test_case, response) in enumerate(
        zip(demo_company_data, responses)
    ):
        assert response.model is not None, (
            f"Expected a model for case {i}: '{test_case['text']}'"
        )
        assert isinstance(response.model, PersonInfo)

        if test_case["expected_name"] is not None:
            assert response.model.name is not None, (
                f"Expected non-null name for case {i}: '{test_case['text']}'"
            )
            assert test_case["expected_name"].lower() in (
                response.model.name.lower()
            ), (
                f"Expected name '{test_case['expected_name']}' "
                f"but got '{response.model.name}' for case {i}:"
                f" '{test_case['text']}'"
            )
        else:
            assert response.model.name is None, (
                f"Expected null name for case {i}, "
                f"got '{response.model.name}'"
            )
