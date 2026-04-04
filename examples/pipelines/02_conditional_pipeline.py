"""
Example: Conditional Pipeline Steps

Demonstrates how to use `condition` on a PipelineStep to skip
steps based on the current input. In this example:

  1. PersonExtractor: extracts person info from text.
  2. BioGenerator: generates a bio, but ONLY if a name was
     extracted (skipped otherwise).

Items without a name skip step 2 and keep their step 1 output.
"""

from agents7.agents import Labeler, Pipeline, PipelineStep
from agents7.models import AgentResponse


def person_to_bio_input(response: AgentResponse) -> dict:
    """Map PersonInfo -> BioGenerator input."""
    person = response.model
    return {
        "name": person.name,
        "age": person.age,
        "occupation": person.occupation,
        "location": person.location,
    }


def has_name(input_data: dict) -> bool:
    """Only run bio generation if a name was extracted."""
    return input_data.get("name") is not None


if __name__ == "__main__":
    extractor = Labeler.from_yaml("../configs/demo_labeler.yml")
    bio_gen = Labeler.from_yaml("../configs/demo_bio_generator.yml")

    pipeline = Pipeline(
        steps=[
            PipelineStep(
                agent=extractor,
                mapper=person_to_bio_input,
            ),
            PipelineStep(
                agent=bio_gen,
                condition=has_name,
            ),
        ]
    )

    test_texts = [
        "Sarah, a 28-year-old data scientist from London.",
        "I bought groceries at the store.",
        "Dr. Chen has been practicing medicine in Singapore.",
    ]

    print(f"Pipeline: {extractor.name} -> {bio_gen.name}")
    print("(Step 2 skipped when no name is extracted)")
    print("=" * 60)

    inputs = [{"text": text} for text in test_texts]
    results = pipeline.act_in_batch(inputs)

    for text, result in zip(test_texts, results):
        print(f"\nInput: {text}")
        if result.success:
            steps_run = len(result.steps)
            print(f"  Steps executed: {steps_run}")
            if hasattr(result.model, "bio"):
                print(f"  Bio: {result.model.bio}")
            else:
                print(f"  Extracted: {result.model}")
        else:
            print("  Pipeline failed")
        print("-" * 60)
