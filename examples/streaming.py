"""
Example: Agent-Level Streaming

Demonstrates how to use `act_stream()` and `act_stream_async()`
to receive LLM output token-by-token, then access the parsed
structured model after iteration completes.
"""

import asyncio

from agents7.agents.labeler import Labeler


if __name__ == "__main__":
    extractor = Labeler.from_yaml("configs/demo_labeler.yml")

    print(f"Agent: {extractor.name} (v{extractor.version})")
    print("=" * 60)

    # --- Sync streaming ---
    print("\n[Sync streaming]")
    print("Tokens: ", end="", flush=True)

    stream = extractor.act_stream(
        {"text": "Sarah, a 28-year-old data scientist from London."}
    )
    for chunk in stream:
        print(chunk, end="", flush=True)

    print("\n")
    response = stream.agent_response
    if response.model:
        print(f"Parsed model: {response.model}")
    print(f"Tokens used: {response.metrics.tokens}")

    # --- Async streaming ---
    print("\n" + "-" * 60)
    print("\n[Async streaming]")

    async def run_async():
        print("Tokens: ", end="", flush=True)
        stream = extractor.act_stream_async(
            {"text": "Dr. Chen practices medicine in Singapore."}
        )
        async for chunk in stream:
            print(chunk, end="", flush=True)

        print("\n")
        response = stream.agent_response
        if response.model:
            print(f"Parsed model: {response.model}")
        print(f"Tokens used: {response.metrics.tokens}")

    asyncio.run(run_async())
