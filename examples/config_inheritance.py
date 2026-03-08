"""
Example: YAML Config Inheritance

Demonstrates how to use `base_config` to share common settings
(engine, provider, model) across multiple agent configs.

The child config (demo_labeler_inherited.yml) inherits from
demo_base.yml and only overrides agent-specific fields like
name, system_prompt, and output_model.
"""

from agents7.agents.labeler import Labeler


if __name__ == "__main__":
    # Load agent that inherits engine config from a base file
    extractor = Labeler.from_yaml("configs/demo_labeler_inherited.yml")

    print(f"Agent: {extractor.name} (v{extractor.version})")
    print(f"Model: {extractor.engine.model_id}")
    print("=" * 60)

    test_texts = [
        "Sarah, a 28-year-old data scientist from London.",
        "Dr. Chen has been practicing medicine in Singapore.",
        "I bought groceries at the store.",
    ]

    inputs = [{"text": text} for text in test_texts]
    responses = extractor.act_in_batch(inputs)

    for text, response in zip(test_texts, responses):
        person = response.model
        print(f"\nInput: {text}")
        print(f"  Name: {person.name}")
        print(f"  Age: {person.age}")
        print(f"  Occupation: {person.occupation}")
        print(f"  Location: {person.location}")
        print("-" * 60)
