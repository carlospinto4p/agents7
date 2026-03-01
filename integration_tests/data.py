"""Shared test data for integration tests."""

TEMPERATURE_DATA = [
    {
        "text": "This is a test text.",
        "expected_reason": "NOT_APPLICABLE",
    },
    {
        "text": "The water in the lake is starting to freeze.",
        "expected_reason": "TOO_COLD",
    },
    {
        "text": ("It's scorching outside; I can barely stay in the sun."),
        "expected_reason": "TOO_HOT",
    },
    {
        "text": ("A light breeze makes the evening feel very pleasant."),
        "expected_reason": "COMFORTABLE",
    },
]

SENTIMENT_DATA = [
    {
        "text": "I absolutely love this! Best day ever!",
        "expected_category": "POSITIVE",
    },
    {
        "text": "This is terrible and I'm very disappointed.",
        "expected_category": "NEGATIVE",
    },
    {
        "text": "The meeting is at 3 PM in conference room B.",
        "expected_category": "NEUTRAL",
    },
]
