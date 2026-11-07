# david_facts

Demonstrates **character facts** — the one framework capability
that survives past the end of a conversation.

## The problem it solves

A `Chatter`'s memory backend stores a transcript keyed by
`conversation_id`. Start a new conversation and that transcript is
gone, so anything the character improvised about itself is gone
with it. Ask David twice what his cat is called and he is free to
answer differently each time, because the second conversation
never saw the first.

Character facts are keyed by the **character** instead. They are
extracted from a finished transcript, stored in a JSON file, and
folded back into the system prompt on every later turn.

## What is wired up here

The whole capability is these four lines of `config.yml`:

```yaml
character_facts:
    path: facts/
    character_id: david
```

`path` is relative to the config file, not to the working
directory. That is deliberate: the store belongs to the agent, so
a web backend and a command-line script both find the same one. An
absolute path is used unchanged.

No extractor engine is declared, so the extractor reuses the
agent's own — `gemma4:e4b` here. Declare one under
`character_facts.extractor.engine` to run extraction on a
different model from the conversation.

## Running it

Needs a local Ollama with `gemma4:e4b`, like the other samples.

```python
from agents7.agents.chatter import Chatter

david = Chatter.from_yaml(
    "samples/david_facts/config.yml"
)
print(david.chat("¿tienes mascota?").response)
david.remember()
```

`remember()` is the whole flow: it reads the finished transcript,
asks the extractor what David said about himself, and appends
whatever is new to `facts/david.json`. Start a fresh `Chatter`
from the same config and the file is already in his prompt.

```json
[
  {
    "text": "Tiene un gato que se llama Pixel",
    "source_conversation": "b1f0c7a2-..."
  }
]
```

## Things worth knowing

- **An empty result is normal.** A conversation that carried
  nothing durable returns no facts, and the file is not created.
  The extractor is told to prefer that over recording a passing
  state like "está esperando a que compile" — see agents7
  v4.20.2, which tightened the prompt after three of four local
  models did exactly that.
- **The files are meant to be edited.** JSON is the only backend
  on purpose. When an extraction records something wrong, open
  `facts/david.json` and fix it.
- **Facts are stored in the character's language**, so what goes
  back into the prompt reads as something David would say rather
  than a translation. David's persona sets `language: Spanish`,
  so his facts are Spanish.
- **`facts/` is gitignored.** Its contents come out of real
  conversations, so nothing in it is committed.
