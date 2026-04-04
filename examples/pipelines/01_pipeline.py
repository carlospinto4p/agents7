"""
Example: Two-Step Pipeline (Extract → Generate Bio)

Demonstrates how to chain two agents using a Pipeline:
  1. PersonExtractor: extracts structured person info from text.
  2. BioGenerator: produces a short biography from the extracted data.

A custom mapper converts the PersonInfo output of step 1 into
the input dict expected by step 2.
"""

from agents7.agents import Labeler, Pipeline, PipelineStep
from agents7.models import AgentResponse


def person_to_bio_input(response: AgentResponse) -> dict:
    """Map PersonInfo → BioGenerator input."""
    person = response.model
    return {
        "name": person.name,
        "age": person.age,
        "occupation": person.occupation,
        "location": person.location,
    }


if __name__ == "__main__":
    # Load agents from YAML configs
    extractor = Labeler.from_yaml("../configs/demo_labeler.yml")
    bio_gen = Labeler.from_yaml("../configs/demo_bio_generator.yml")

    # Build the pipeline
    pipeline = Pipeline(
        steps=[
            PipelineStep(
                agent=extractor,
                mapper=person_to_bio_input,
            ),
            PipelineStep(agent=bio_gen),
        ]
    )

    # Test cases
    test_texts = [
        "Sarah, a 28-year-old data scientist from London.",
        "Dr. Chen has been practicing medicine in Singapore.",
        "I bought groceries at the store.",
    ]

    print(f"Pipeline: {extractor.name} → {bio_gen.name}")
    print("=" * 60)

    # --- Single mode ---
    print("\n[Single mode]")
    for text in test_texts:
        result = pipeline.act({"text": text})
        print(f"\nInput: {text}")
        if result.success:
            person = result.steps[0].model
            print(f"  Extracted: {person}")
            print(f"  Bio: {result.model.bio}")
        else:
            print("  Pipeline failed (could not extract info)")
        print("-" * 60)

    # --- Batch mode ---
    print("\n[Batch mode]")
    inputs = [{"text": text} for text in test_texts]
    results = pipeline.act_in_batch(inputs)

    for text, result in zip(test_texts, results):
        print(f"\nInput: {text}")
        if result.success:
            print(f"  Bio: {result.model.bio}")
        else:
            print("  Pipeline failed")
        print("-" * 60)
