"""
Example: Roleplay Agent with a persona-driven Chatter

Demonstrates how to create a character with a structured
persona. David is a Spanish journalist who always responds
in Spanish, staying in character across all questions.

A `persona:` section in the YAML auto-generates the system
prompt — no dedicated subclass is needed.
"""

from agents7.agents import Chatter


def _ask(agent, question: str) -> None:
    """Ask a question and print the response with timing."""
    print(f"\nYou: {question}")
    response = agent.chat(question)
    duration = response.metrics.total_duration
    tokens = response.metrics.tokens
    print(f"David: {response.response}")
    print(f"  ({tokens} tokens, {duration:.1f}s)")


if __name__ == "__main__":
    david = Chatter.from_yaml("../configs/demo_personality_chatter.yml")

    print(f"Agent: {david.name} (v{david.version})")
    print(f"Model: {david.engine.model_id}")
    print(f"Persona: {david.persona.full_name}")
    print(f"Language: {david.persona.language}")
    print("=" * 60)

    # --- Biographical questions ---
    print("\n[Biographical questions]")

    _ask(david, "What's your name and where are you from?")
    _ask(david, "When were you born?")
    _ask(david, "What do you do for a living?")
    _ask(david, "Tell me about your neighborhood.")

    # --- Physical description ---
    print("\n" + "=" * 60)
    print("\n[Physical description]")

    _ask(david, "Describe yourself physically.")

    # --- Personal interests ---
    print("\n" + "=" * 60)
    print("\n[Interests and personality]")

    _ask(david, "What are your hobbies?")
    _ask(david, "What's your favorite food?")
    _ask(david, "Do you follow any football team?")

    # --- Consistency check ---
    print("\n" + "=" * 60)
    print("\n[Consistency check - asking again]")

    _ask(david, "Wait, what was your name again?")

    print(f"\nTotal turns: {david.turns}")
    print(f"History: {len(david.history)} messages")
