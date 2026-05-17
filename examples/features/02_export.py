"""
Example: Exporting Agent Responses

Demonstrates how to export agent responses to JSONL and CSV
files using the built-in export helpers.
"""

from agents7.agents.labeler import Labeler
from agents7.eval.export import export_csv, export_jsonl


if __name__ == "__main__":
    extractor = Labeler.from_yaml("../configs/demo_labeler.yml")

    test_texts = [
        "Sarah, a 28-year-old data scientist from London.",
        "Dr. Chen has been practicing medicine in Singapore.",
        "My neighbor David, 45, works as a carpenter in Madrid.",
    ]

    print(f"Agent: {extractor.name} (v{extractor.version})")
    print("=" * 60)

    # Process all inputs
    inputs = [{"text": text} for text in test_texts]
    responses = extractor.act_in_batch(inputs)

    # Export to JSONL (one JSON object per line)
    export_jsonl(responses, "demo_output.jsonl")
    print("Exported to demo_output.jsonl")

    # Export to CSV (auto-detects model fields)
    export_csv(responses, "demo_output.csv")
    print("Exported to demo_output.csv")

    # Export to CSV with specific fields only
    export_csv(
        responses,
        "demo_output_names.csv",
        fields=["name", "occupation"],
    )
    print("Exported to demo_output_names.csv (name + occupation only)")

    # Show what was exported
    print("\n" + "-" * 60)
    print("\nJSONL preview:")
    with open("demo_output.jsonl") as f:
        for line in f:
            print(f"  {line.strip()}")

    print("\nCSV preview:")
    with open("demo_output.csv") as f:
        for line in f:
            print(f"  {line.strip()}")
