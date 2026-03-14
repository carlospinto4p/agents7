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
    # NOT_APPLICABLE is disabled on the Sentiment model
    # (NOT_APPLICABLE = False) since NEUTRAL covers that case.
]

PIPELINE_DATA = [
    {
        "text": "John is a 35-year-old engineer living in Berlin.",
        "expected_name": "John",
    },
    {
        "text": "Maria is a CEO who recently moved to Tokyo.",
        "expected_name": "Maria",
    },
]

COMPANY_DATA = [
    {
        "text": ("John is a 35-year-old engineer at Acme Corp in Berlin."),
        "expected_name": "John",
    },
    {
        "text": "Sara is an intern at Acme Corp headquarters.",
        "expected_name": "Sara",
    },
    {
        "text": ("Maria works at a different company as a designer."),
        "expected_name": None,
    },
]

SIMPLE_TEMPERATURE_CASES = [
    {
        "text": "It is -10 degrees outside.",
        "expected_reason": "TOO_COLD",
    },
    {
        "text": "The temperature is 35 degrees in the shade.",
        "expected_reason": "TOO_HOT",
    },
    {
        "text": "She bought a new book.",
        "expected_reason": "NOT_APPLICABLE",
    },
]
