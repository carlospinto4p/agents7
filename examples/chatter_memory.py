"""
Example: Persistent Memory with Chatter

Demonstrates how to use memory backends to persist
conversation history across sessions. Also shows how
conversation_id isolates different users sharing the
same memory backend.

Run this script twice to see history persistence:
  1st run: starts fresh conversations for Alice and Bob
  2nd run: resumes where they left off
"""

import os

from agents7.agents.chatter import Chatter
from agents7.agents.memory import JsonFileMemory
from agents7.llm import LLMEngine


MEMORY_FILE = "chatter_history.json"


def fmt_metrics(response) -> str:
    """Format token count and duration."""
    m = response.metrics
    return f"({m.tokens} tokens, {m.total_duration:.1f}s)"


def make_chatter_for_user(
    user_id: str,
    memory: JsonFileMemory,
) -> Chatter:
    """Create a Chatter instance for a specific user."""
    engine = LLMEngine("ollama", "qwen3")
    return Chatter(
        name="FriendlyBot",
        version=1,
        engine=engine,
        system_prompt=(
            "You are a friendly assistant. Keep responses "
            "short (1-2 sentences). If you remember prior "
            "conversation, reference it naturally."
        ),
        temperature=0.7,
        memory=memory,
        conversation_id=user_id,
    )


if __name__ == "__main__":
    memory = JsonFileMemory(MEMORY_FILE)
    is_first_run = not os.path.exists(MEMORY_FILE)

    print("=" * 60)
    if is_first_run:
        print("FIRST RUN — starting fresh conversations")
    else:
        print("RESUMING — loading previous conversations")
        convs = memory.list_conversations()
        print(f"Found conversations: {convs}")
    print("=" * 60)

    # --- Alice's conversation ---
    print("\n[Alice]")
    alice = make_chatter_for_user("alice", memory)

    if is_first_run:
        print(f"  History: {len(alice.history)} messages")
        response = alice.chat("Hi! My name is Alice.")
        print("  Alice: Hi! My name is Alice.")
        print(f"  Bot:   {response.response}")
        print(f"         {fmt_metrics(response)}")

        response = alice.chat("I'm a data scientist working on NLP.")
        print("  Alice: I'm a data scientist working on NLP.")
        print(f"  Bot:   {response.response}")
        print(f"         {fmt_metrics(response)}")
    else:
        print(f"  History: {len(alice.history)} messages")
        response = alice.chat("What do you remember about me?")
        print("  Alice: What do you remember about me?")
        print(f"  Bot:   {response.response}")
        print(f"         {fmt_metrics(response)}")

    # --- Bob's conversation (isolated from Alice) ---
    print("\n[Bob]")
    bob = make_chatter_for_user("bob", memory)

    if is_first_run:
        print(f"  History: {len(bob.history)} messages")
        response = bob.chat("Hey, I'm Bob. I'm learning to cook.")
        print("  Bob: Hey, I'm Bob. I'm learning to cook.")
        print(f"  Bot: {response.response}")
        print(f"       {fmt_metrics(response)}")
    else:
        print(f"  History: {len(bob.history)} messages")
        response = bob.chat("What do you remember about me?")
        print("  Bob: What do you remember about me?")
        print(f"  Bot: {response.response}")
        print(f"       {fmt_metrics(response)}")

    # --- Show stored conversations ---
    print("\n" + "=" * 60)
    print(f"Conversations in {MEMORY_FILE}:")
    for conv_id in memory.list_conversations():
        msgs = memory.load(conv_id)
        print(f"  {conv_id}: {len(msgs)} messages")

    if not is_first_run:
        print(f"\nCleaning up {MEMORY_FILE}...")
        os.remove(MEMORY_FILE)
        print("Run the script again for a fresh start.")
    else:
        print("\nRun this script again to see persistence!")
