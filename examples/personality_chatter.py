"""
Example: Roleplay Agent with PersonalityChatter

Demonstrates how to create a character with a structured
persona. David is a Spanish journalist who always responds
in Spanish, staying in character across all questions.
"""

from agents7.agents.personality import PersonalityChatter


if __name__ == "__main__":
    david = PersonalityChatter.from_yaml(
        "configs/demo_personality_chatter.yml"
    )

    print(f"Agent: {david.name} (v{david.version})")
    print(f"Persona: {david.persona.full_name}")
    print(f"Language: {david.persona.language}")
    print("=" * 60)

    # --- Biographical questions ---
    print("\n[Biographical questions]")

    questions = [
        "What's your name and where are you from?",
        "When were you born?",
        "What do you do for a living?",
        "Tell me about your neighborhood.",
    ]

    for q in questions:
        print(f"\nYou: {q}")
        response = david.chat(q)
        print(f"David: {response.response}")

    # --- Physical description ---
    print("\n" + "=" * 60)
    print("\n[Physical description]")

    print("\nYou: Describe yourself physically.")
    response = david.chat("Describe yourself physically.")
    print(f"David: {response.response}")

    # --- Personal interests ---
    print("\n" + "=" * 60)
    print("\n[Interests and personality]")

    interest_questions = [
        "What are your hobbies?",
        "What's your favorite food?",
        "Do you follow any football team?",
    ]

    for q in interest_questions:
        print(f"\nYou: {q}")
        response = david.chat(q)
        print(f"David: {response.response}")

    # --- Consistency check ---
    print("\n" + "=" * 60)
    print("\n[Consistency check - asking again]")

    print("\nYou: Wait, what was your name again?")
    response = david.chat("Wait, what was your name again?")
    print(f"David: {response.response}")

    print(f"\nTotal turns: {david.turns}")
    print(f"History: {len(david.history)} messages")
