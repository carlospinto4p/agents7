"""
Example: Contextual Roleplay with SceneChatter

Demonstrates how to place a character in a specific scene.
Old Thomas is a medieval tavern keeper trying to close up
for the night. He has no knowledge of anything modern, is
suspicious of strangers, and won't serve anyone who can't
pay.
"""

from agents7.agents.scene import SceneChatter


def _ask(agent, question: str) -> None:
    """Ask a question and print the response with timing."""
    print(f"\nYou: {question}")
    response = agent.chat(question)
    duration = response.metrics.total_duration
    tokens = response.metrics.tokens
    print(f"Thomas: {response.response}")
    print(f"  ({tokens} tokens, {duration:.1f}s)")


if __name__ == "__main__":
    thomas = SceneChatter.from_yaml("../configs/demo_scene_chatter.yml")

    print(f"Agent: {thomas.name} (v{thomas.version})")
    print(f"Model: {thomas.engine.model_id}")
    print(f"Persona: {thomas.persona.full_name}")
    print(f"Scene: {thomas.scene.setting}")
    print("=" * 60)

    # --- Entering the tavern ---
    print("\n[Entering the tavern]")

    _ask(thomas, "Good evening! Is this place still open?")
    _ask(thomas, "I'd like some ale and whatever food you have.")

    # --- Testing scene rules ---
    print("\n" + "=" * 60)
    print("\n[Testing scene rules]")

    _ask(thomas, "Can I charge my phone somewhere?")
    _ask(thomas, "What do you think about democracy?")

    # --- Interacting with the scene ---
    print("\n" + "=" * 60)
    print("\n[Scene awareness]")

    _ask(thomas, "Who's that hooded figure in the corner?")
    _ask(thomas, "The sailor at the bar looks like trouble.")

    # --- Pushing the goal ---
    print("\n" + "=" * 60)
    print("\n[Character goal]")

    _ask(thomas, "I think I'll stay a while longer.")

    print(f"\nTotal turns: {thomas.turns}")
    print(f"History: {len(thomas.history)} messages")
