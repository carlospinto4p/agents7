"""
Example: Sentiment Classification with Classifier

Demonstrates how to use a Classifier to analyze text sentiment
and classify it into predefined categories with justifications.
"""

from agents7.agents.classifier import Classifier


if __name__ == "__main__":
    # Load the sentiment classifier from config
    classifier = Classifier.from_yaml("configs/demo_sentiment_classifier.yml")

    # Test cases demonstrating different sentiment scenarios
    test_texts = [
        "I can't believe how amazing this experience was!",
        "This is the worst customer service I've ever encountered.",
        "The report contains the quarterly sales figures.",
        "I'm so grateful for all the support you've given me.",
        "Nothing works properly and I'm extremely frustrated.",
        "The sky is blue and the grass is green.",
        "I hope everything works out for the best!",
    ]

    print(f"Agent: {classifier.name} (v{classifier.version})")
    print("=" * 60)

    for text in test_texts:
        response = classifier.act({"text": text})
        sentiment = response.model

        print(f"\nInput: {text}")
        print("Classification:")
        print(f"  Category: {sentiment.category.name}")
        print(f"  Reason: {sentiment.reason.name}")
        print("-" * 60)
