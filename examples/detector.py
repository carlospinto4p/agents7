"""
Example: Temperature Detection with Detector

Demonstrates how to use a Detector to extract temperature-related
properties from text descriptions.
Uses batch processing for efficient handling of multiple inputs.
"""

from agents7.agents.detector import Detector


if __name__ == "__main__":
    # Load the temperature detector from config
    detector = Detector.from_yaml("configs/demo_temperature_detector.yml")

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

    print(f"Agent: {detector.name} (v{detector.version})")
    print("=" * 60)

    # Use batch processing for multiple inputs
    inputs = [{"text": text} for text in test_texts]
    responses = detector.act_in_batch(inputs)

    for text, response in zip(test_texts, responses):
        temp = response.model

        print(f"\nInput: {text}")
        print("Detected:")
        print(f"  Value: {temp.value}")
        print(f"  Reason: {temp.reason.name}")
        print("-" * 60)
