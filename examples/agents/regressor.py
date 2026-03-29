"""
Example: Temperature Regression with Regressor

Demonstrates how to use a Regressor to extract numeric
temperature values from text descriptions.
Uses batch processing for efficient handling of multiple inputs.
"""

from agents7.agents.regressor import Regressor


if __name__ == "__main__":
    # Load the temperature regressor from config
    # (Regressor extends Detector, so it reuses the same YAML)
    regressor = Regressor.from_yaml(
        "../configs/demo_temperature_detector.yml"
    )

    # Test cases demonstrating different temperature scenarios
    test_texts = [
        "This is a test text.",
        "The water in the lake is starting to freeze.",
        "It's scorching outside; I can barely stay in the sun.",
        "A light breeze makes the evening feel very pleasant.",
        "The heater is on full blast in the room.",
        "I walked outside and shivered immediately.",
        "The coffee was lukewarm when I drank it.",
        "Summer in this city often exceeds 35 degrees.",
        "She just bought a new pair of shoes.",
        "The snow on the mountains hasn't melted yet.",
        "Perfect weather for a picnic: neither too hot nor too cold.",
    ]

    print(f"Agent: {regressor.name} (v{regressor.version})")
    print("=" * 60)

    # Use batch processing for multiple inputs
    inputs = [{"text": text} for text in test_texts]
    responses = regressor.act_in_batch(inputs)

    for text, response in zip(test_texts, responses):
        temp = response.model

        print(f"\nInput: {text}")
        print("Detected:")
        print(f"  Value: {temp.value}")
        print(f"  Reason: {temp.reason.name}")
        print("-" * 60)
