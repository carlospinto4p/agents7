"""
Example: Multi-Turn Conversation with Chatter

Demonstrates how to use a Chatter agent for conversational
interactions. Shows basic chat, streaming, and history
management.
"""

from agents7.agents.chatter import Chatter


if __name__ == "__main__":
    chatter = Chatter.from_yaml("configs/demo_chatter.yml")

    print(f"Agent: {chatter.name} (v{chatter.version})")
    print("=" * 60)

    # --- Basic multi-turn chat ---
    print("\n[Multi-turn conversation]")

    messages = [
        "Hi! What's your name?",
        "Can you explain what a neural network is in one sentence?",
        "Now explain it like I'm five years old.",
    ]

    for msg in messages:
        print(f"\nYou: {msg}")
        response = chatter.chat(msg)
        print(f"Bot: {response.response}")
        print(f"  ({response.metrics.tokens} tokens)")

    print(f"\nHistory length: {len(chatter.history)} messages")

    # --- Streaming ---
    print("\n" + "=" * 60)
    print("\n[Streaming response]")

    msg = "Tell me a very short joke."
    print(f"\nYou: {msg}")
    print("Bot: ", end="", flush=True)

    stream = chatter.chat_stream(msg)
    for chunk in stream:
        print(chunk, end="", flush=True)

    print(f"\n  ({stream.chat_response.metrics.tokens} tokens)")

    # --- Reset and new conversation ---
    print("\n" + "=" * 60)
    print("\n[After reset]")

    chatter.reset()
    print(f"History length: {len(chatter.history)} messages")

    response = chatter.chat("Do you remember what we talked about?")
    print("\nYou: Do you remember what we talked about?")
    print(f"Bot: {response.response}")
