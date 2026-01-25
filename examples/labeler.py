"""
Example: Entity Extraction with Labeler

Demonstrates how to use a Labeler to extract structured information
(person details) from unstructured text descriptions.
"""

from agents7.agents.labeler import Labeler


if __name__ == "__main__":
    # Load the person extractor labeler from config
    extractor = Labeler.from_yaml("configs/demo_labeler.yml")

    # Test cases demonstrating different extraction scenarios
    test_texts = [
        "Sarah, a 28-year-old data scientist from London, loves hiking.",
        "The new intern started today. His name is Alex.",
        "Dr. Chen has been practicing medicine in Singapore for 20 years.",
        "I bought groceries at the store.",
        "My neighbor David, 45, works as a carpenter in Madrid.",
    ]

    print(f"Agent: {extractor.name} (v{extractor.version})")
    print("=" * 60)

    for text in test_texts:
        response = extractor.act({"text": text})
        person = response.model

        print(f"\nInput: {text}")
        print("Extracted:")
        print(f"  Name: {person.name}")
        print(f"  Age: {person.age}")
        print(f"  Occupation: {person.occupation}")
        print(f"  Location: {person.location}")
        print("-" * 60)
