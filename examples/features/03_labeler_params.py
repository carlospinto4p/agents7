"""
Example: Entity Extraction with Prompt Parameters

Demonstrates how to use prompt_params to inject dynamic values
into an agent's system prompt. Shows two approaches:
1. Static params via YAML configuration.
2. Callable params via Python instantiation (e.g., current date).
"""

from datetime import date

from agents7.agents.labeler import Labeler


def example_from_yaml():
    """Load agent from YAML with static prompt_params."""
    extractor = Labeler.from_yaml("../configs/demo_labeler_params.yml")

    print(f"Agent: {extractor.name} (v{extractor.version})")
    print(f"Params: {extractor.prompt_params}")
    print("=" * 60)

    test_texts = [
        "Sarah, 28, just joined Acme Corp as a data scientist in London.",
        "Dr. Chen has been practicing medicine in Singapore.",
        "Javier, 45, is a senior engineer at Acme Corp in Madrid.",
        "I bought groceries at the store.",
    ]

    inputs = [{"text": text} for text in test_texts]
    responses = extractor.act_in_batch(inputs)

    for text, response in zip(test_texts, responses):
        person = response.model
        print(f"\nInput: {text}")
        print("Extracted:")
        print(f"  Name: {person.name}")
        print(f"  Age: {person.age}")
        print(f"  Occupation: {person.occupation}")
        print(f"  Location: {person.location}")
        print("-" * 60)


def example_with_callable():
    """
    Create agent in Python with a callable prompt_param.
    The current_date param is evaluated fresh each time
    the prompt is accessed.
    """
    extractor = Labeler.from_yaml("../configs/demo_labeler_params.yml")

    # Override with callable params via Python instantiation
    extractor = Labeler(
        name=extractor.name,
        version=extractor.version,
        engine=extractor.engine,
        system_prompt=extractor.system_prompt,
        output_model=extractor.output_model,
        examples=extractor.examples,
        prompt_params={
            "company": "Acme Corp",
            "current_date": lambda: str(date.today()),
        },
    )

    print(f"\nAgent: {extractor.name} (callable params)")
    print("Resolved prompt starts with:")
    # Show the first 3 lines of the resolved prompt
    for line in extractor.resolved_system_prompt.splitlines()[:3]:
        print(f"  {line}")
    print("=" * 60)

    response = extractor.act(
        {"text": "Alice, 30, is a manager at Acme Corp in Tokyo."}
    )
    person = response.model
    print(f"  Name: {person.name}")
    print(f"  Age: {person.age}")
    print(f"  Occupation: {person.occupation}")
    print(f"  Location: {person.location}")


if __name__ == "__main__":
    example_from_yaml()
    example_with_callable()
