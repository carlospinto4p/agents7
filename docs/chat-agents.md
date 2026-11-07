# Chat agents

## Persona-Driven Chatter (Roleplay Agent)

Give a `Chatter` a structured `Persona` for consistent character
roleplay. The system prompt is auto-generated from biographical
data. `persona` is an optional plugin on the base `Chatter` (the
former `PersonalityChatter` subclass was removed in v4.0.0):

```python
from agents7.agents import Chatter, Persona, PhysicalTraits

persona = Persona(
    full_name="David García",
    birthdate="1990-12-04",
    nationality="Spanish",
    occupation="journalist",
    physical=PhysicalTraits(
        height="178 cm",
        build="medium",
        hair="dark brown, short",
        eyes="brown",
    ),
    personality=["curious", "empathetic", "direct"],
    backstory="Grew up in Madrid, studied journalism at UCM.",
    interests=["politics", "football", "cooking"],
    speaking_style="casual and direct",
)

agent = Chatter(
    name="david",
    version=1,
    engine=engine,
    system_prompt="",
    persona=persona,
    extra_instructions="Always respond in Spanish.",
)

response = agent.chat("¿Cómo te llamas?")
```

Every `Chatter` reply is returned and stored without empty lines.
A persona that writes "two or three short lines" gets them as
consecutive lines; a model that separates them with an empty line
would otherwise hand a chat client an empty message between them.
A live stream still yields chunks as generated, and its
`chat_response` carries the cleaned reply.

YAML configuration:

```yaml
name: david
version: 1
engine:
    provider: anthropic
    model: claude-sonnet-4-20250514
persona:
    full_name: David García
    birthdate: "1990-12-04"
    nationality: Spanish
    occupation: journalist
    physical:
        height: 178 cm
        build: medium
        hair: dark brown, short
        eyes: brown
    personality:
        - curious
        - empathetic
        - direct
    backstory: |
        Grew up in Madrid, studied journalism at UCM.
    interests:
        - politics
        - football
        - cooking
    speaking_style: casual and direct
extra_instructions: |
    Always respond in Spanish.
```

The persona includes: name, birthdate, nationality, occupation, physical traits, personality adjectives, backstory, interests, speaking style, and a freeform `extra` dict for any additional details.

## Scene-Driven Chatter (Contextual Roleplay Agent)

Add a `Scene` alongside the persona to place the character in a
specific **scene**: a location, situation, time period, and set
of behavioral rules. The system prompt frames everything as a
role-play scenario. A `scene` requires a `persona` (the former
`SceneChatter` subclass was removed in v4.0.0):

```python
from agents7.agents import Chatter, Persona, Scene

persona = Persona(
    full_name="Old Thomas",
    occupation="tavern keeper",
    personality=["gruff", "practical", "shrewd"],
)
scene = Scene(
    setting="A medieval tavern at night",
    situation="The tavern is about to close.",
    time_period="Medieval England, 14th century",
    rules=[
        "No knowledge of anything after the 14th century.",
        "Suspicious of strangers.",
    ],
    goal="Close the tavern and get everyone out.",
)
agent = Chatter(
    name="thomas",
    version=1,
    engine=engine,
    system_prompt="",
    persona=persona,
    scene=scene,
)

response = agent.chat("Good evening!")
```

The scene includes: setting, situation, time period, rules, other characters present, goal, and a freeform `extra` dict. See `docs/chatter.md` for the full YAML schema.

## Splitting a Roleplay Agent into Character and Chat Goal

A roleplay agent has two independent axes, and putting them in one
file ties them together: rename the character and you edit a file
full of scene rules; reuse the character in a different
conversation and you copy its whole scenario with it.

`character` and `chat_goal` each name a YAML file, and the loader
folds both into the config:

```yaml
# config.yml — the deployment: model, prices, stores, guard
character: characters/david.yml
chat_goal: goals/source_interview.yml
name: david
version: 1
agent_type: guarded_scene_chatter
engine:
    provider: openai_compat
    model: deepseek-chat
```

```yaml
# characters/david.yml — who he is. No scene, ever.
persona:
    full_name: David García López
    language: Spanish
    speaking_style: |
        Casual y directo. Frases cortas.
extra_instructions: |
    ## Cómo escribe
    Usuario: ¿qué tal?
    $character: liado con un artículo, como siempre
```

```yaml
# goals/source_interview.yml — what he is doing. No persona, ever.
scene:
    setting: Un café ruidoso en Malasaña, por la tarde.
    rules:
        - Nunca revelas quién es tu fuente.
    goal: Conseguir detalles verificables para publicar la historia.
can_open: true
extra_instructions: |
    ## Cómo trabaja
    Fuente: ¿vas a publicar mi nombre?
    $character: no, eso no sale de aquí
```

Three rules govern the composition:

- **Neither file may hold the other's section.** A `character`
  declaring `scene`, or a `chat_goal` declaring `persona`, raises
  `AgentConfigurationError`. The split is enforced rather than
  suggested, because a config that has been split can otherwise
  grow quietly back into one file and the swap stops working with
  nothing failing.
- **`extra_instructions` and `opening_instructions` accumulate**,
  in a fixed order: character, then chat goal, then whatever the
  agent config says itself. Both axes have something to say about
  how the agent writes, so neither silently replaces the other.
  Every other key is an ordinary deep merge with the agent config
  winning, so a deployment can pin what the fragments left open.
- **`$character` resolves to the persona's `full_name`.** That is
  what lets a chat-goal file write a worked example without saying
  who is in it. Swap `characters/david.yml` for another one and every
  example in the goal reads correctly, including the ones the
  guard sees.

Fragments are one level deep: neither may declare `base_config`,
`character`, or `chat_goal` of its own. `base_config` inheritance
still works on the agent config itself, and resolves first.

To read the composed result without building the agent, use
`agents7.agents.resolve_config(path)`. A config-contract test or a
catalog listing needs this.

## Opening the Conversation (Agent Speaks First)

`chat()` answers a message. A character whose persona says how it
greets a stranger needs the other call: one that produces a first
turn with nothing to answer. Set `can_open=True` and use
`open_chat()`:

```python
agent = Chatter(
    name="david",
    version=1,
    engine=engine,
    system_prompt="",
    persona=persona,
    can_open=True,
)

opening = agent.open_chat()   # the agent writes first
print(opening.response)       # "¡Hola! ¿Qué tal la tarde?"

agent.chat("Bien, ¿y tú?")    # the conversation continues normally
```

Without `can_open=True` the call raises
`AgentConfigurationError` rather than inventing an opening the
agent was never given. The async and streaming siblings are
`open_chat_async()`, `open_chat_stream()`, and
`open_chat_stream_async()`; on `GuardedChatter` the opening turn
runs through the guard like any other.

[Per-conversation context](agents.md#per-conversation-context) reaches the
opening turn like any other. That matters most here, because the
first line is written before the other person has said anything,
so `set_context` is all the agent has to go on:

```python
agent.set_context(nick="lector42")
agent.open_chat()   # greets knowing who it is writing to
```

The model is asked for that first message by an instruction sent
as a user message, which is **not** stored. The transcript
begins with the agent's own line. Set `opening_prompt` to replace
the built-in English instruction, which is what a persona writing
in another language usually wants:

```yaml
can_open: true
opening_prompt: "[Escribe el primer mensaje de esta conversación.]"
```

### Where the opening lines themselves go

`opening_prompt` is the instruction that asks for a first message.
Guidance on *how* the character opens, whether that is its written
opening lines, their length or their register, goes in
`opening_instructions`,
which is appended to the system prompt on the opening turn and on
no other:

```yaml
can_open: true
opening_instructions: |
    Cuando le toca abrir a él, siempre muy corto: una línea y para.

    $character: ¡Buenas! ¿Qué tal?
    $character: Hola, ¿tienes un minuto?
```

`$character` resolves to the persona's `full_name` here as it
does in the standing prompt, so the same lines can be written
once and read under any character.

Put those lines in `extra_instructions` instead and they are in
front of the model on every single turn. A bank of opening
lines the model can read on every turn is a bank it can copy on
every turn. The failure that produced this field: a character who
had just said goodnight answered the other person's *"vale"* with
*"¡Buenas! ¿Qué tal?"*, the first line of its own opening
section, fifteen turns into the conversation.

On `GuardedChatter` a rejected opening turn regenerates from the
same messages, so the retry still knows it is opening.

## Character Facts (Memory Across Conversations)

A `MemoryBackend` stores the *transcript* under a
`conversation_id`, so anything a character improvises about
itself dies with that conversation. A **fact store** keys what
the character said to the *character* instead. Recorded facts are
folded into the system prompt on every turn, in every later
conversation.

`remember()` runs a `FactExtractor` over the finished transcript
and writes what it found. It makes an LLM call, so call it when a
conversation is over, not inside a chat turn:

```python
from agents7.agents import (
    Chatter,
    FactExtractor,
    JsonFileFactStore,
    Persona,
)

store = JsonFileFactStore("facts/")

agent = Chatter(
    name="david",
    version=1,
    engine=engine,
    system_prompt="",
    persona=Persona(full_name="David García López", language="Spanish"),
    facts=store,
    character_id="david",
    fact_extractor=FactExtractor(
        name="david-facts", version=1, engine=engine
    ),
)

agent.chat("¿Cuál es tu película favorita?")
# → "El secreto de sus ojos, sin duda."

agent.remember()
# [CharacterFact(text='Su película favorita es El secreto de sus
#                ojos', source_conversation='default')]

# A different conversation, later — same character:
later = Chatter(..., facts=store, character_id="david")
later.chat("¿Te gusta el cine?")
# David answers as somebody whose favourite film is already
# El secreto de sus ojos.
```

The extractor records only what the **character** said about
itself, only when it is durable, and only when it is not already
known. It ignores what the other person said, and passing states
("estoy cansado", the weather).

**Facts are stored in the character's own language, not in
English.** The language comes from `persona.language`. David
declares `Spanish`, so his facts are stored in Spanish even when
the other person writes in English. That keeps the system prompt
in one language throughout, so the facts read as things David
would say rather than as English notes about him. Proper nouns
are kept as the character wrote them: a film he called *El
secreto de sus ojos* is not rendered as *The Secret in Their
Eyes*. Without a persona,
or without a declared language, the extractor follows whatever
language the transcript is in.

Facts are stored as one JSON file per character, so an entry that
comes out wrong can be corrected by hand:

```json
[
  {
    "text": "Su película favorita es El secreto de sus ojos",
    "source_conversation": "default"
  }
]
```

Other methods and fields:

- `agent.character_facts`: the facts currently in the prompt.
- `agent.forget()`: erase every fact for this character.
  `reset()` clears the conversation and leaves the facts alone.
- `store.add(character_id, ["..."])`: write a fact by hand,
  without an LLM call.
- `character_id` defaults to the agent's `name`, so several
  agents can share one character by setting it explicitly.
- `banned_fact_phrases`: phrases the character must never say.
  A fact containing one is not written by `remember()`, and one
  already in the file is left out of the prompt with a warning
  naming it. The file itself is not edited. Matching ignores
  case and spacing.

A stored fact outlives the config it was written under. Without a
ban list, a phrase the persona forbids today can come straight
back from a fact recorded before the rule existed.

YAML configuration works on `Chatter`, `GuardedChatter`, and
`GuardedSceneChatter`. The extractor reuses the agent's engine
unless you name another:

```yaml
character_facts:
    path: facts/
    character_id: david
    banned_phrases:
        - sin remedio
    extractor:
        engine:
            provider: ollama
            model: gemma3
```

A relative `path` is resolved against **the directory holding the
config file**, not the directory the process runs from. The folder
above lands beside the YAML that declared it, so a web backend and
a command-line script loading the same config read and write the
same facts. Give an absolute path to put the store somewhere else.
`memory.path` is anchored the same way.

## Scoring a Chat Agent's Style

A conversational agent has neither a label nor a tool call to
score. Two good replies to "how are you" share no wording. So the only
instrument it had was a person reading a transcript, which is slow,
subjective, and blind to the half of a persona that is a plain
**string property of the output**: message count, line length,
forbidden words, how a reply opens, whether two replies in a row
end on a question.

A **style contract** scores that half with no model and no person.
What is not checkable that way, such as whether the agent is good
company or the tone is right, stays with the human reading the
transcript. That is the point: the read gets spent on the questions only a person can
answer.

A third thing comes out of that read and is neither: **the
operator's correction of a specific line, and the rule it becomes.**
`eval.chat_feedback` is the format for it. One entry carries a
rewrite, a rule, or both, since a real correction is usually both.
Three things are derived from the one file: `rule_changes()` groups
the rules by the config file they land in and flags the ones that
have not shipped, `eval_rows()` turns corrected turns into
machine-readable targets, and `render_feedback()` writes the
operator's report. `agents7 chat-feedback <file>` does all three.
See **Feedback on a chat agent** below.

Two things differ from the other scorers:

- **The counterpart is scripted, deliberately.** Judging whether an
  agent is good company needs a *live* counterpart, since a reply
  written before the agent spoke proves nothing. Detecting that a
  rule stopped holding needs the opposite: the same stimulus every
  time, so the only thing that changed is the config.
- **A rule scores as a rate, not a pass.** Chat agents run at
  temperature, so one run of one script is an anecdote. Each rule
  declares the rate it tolerates via `budget`: `0.0` is a hard rule,
  `0.5` tolerates a tendency, and `None` measures without ever
  failing. `None` is for a number worth watching that has no
  agreed threshold yet.

The contract lives beside the agent as data, like a golden set:

```yaml
repeats: 5
rules:
    - id: one_message_when_dry
      description: When the user writes three words or fewer, one message.
      check: max_messages
      args: {limit: 1, when_prompt_words_at_most: 3}
      budget: 0.0
    - id: ends_on_a_question
      description: Asking is one move, not the default.
      check: ends_with_question
      budget: null           # measured, never failed
scripts:
    - id: cold
      opens: true
      lines: ["bien", "normal", "ya", "no"]
```

```python
from agents7.eval import load_style_contract, run_style_contract

contract = load_style_contract("data/style_contract.yml")
report = run_style_contract(lambda: build_agent(), contract)
print(report.summary())
assert report.passed
```

**A script can hand the agent context.** A rule that depends on a
[context param](agents.md#per-conversation-context), such as the other
person's nickname, cannot be exercised by a script that leaves every
param at its default. A script's optional `context` mapping is
passed to `set_context` before its first turn:

```yaml
scripts:
    - id: nick
      opens: true
      context: {nick: Marta_87}
      lines: ["hola"]
```

The context is part of the stimulus, so it is part of the run's
fingerprint: changing it starts a fresh baseline. A script that
declares none hashes exactly as it did before.

**Use `run_style_contract_async` for anything real.** The
conversations are independent, because each gets its own agent
from the factory and shares no state. Running them one after
another costs wall clock and buys nothing. Turns *within* a conversation
stay ordered either way, so the report is identical run for run,
labels included:

```python
from agents7.eval import run_style_contract_async

report = await run_style_contract_async(
    build_agent, contract, concurrency=5
)
```

This is what decides whether a contract gets run at all. On a
chat agent, 20 conversations took about 17 minutes serially, which
is too slow to put on every version. The ceiling on `concurrency`
is the provider's rate limit, not the machine, because every one
of these is waiting on a network call.

The async runner closes every agent it built (`aclose()`, see
[Closing Async Clients](providers.md#closing-async-clients)) before it returns,
so an `asyncio.run` of it ends without an *"Event loop is closed"*
traceback. The agents stay usable; their next async call connects
afresh.

**A conversation that raises loses only itself.** Both runners
catch it, leave that conversation unscored, and score the rest.
A provider dropping one connection costs one conversation rather
than the whole invocation. On a chat agent it used to cost
about seventeen minutes and seventeen cents, with nothing
recorded. The runs that did not finish are in `report.lost`, are
named in the summary, and `report.complete` says whether there
were any. Their replies are **not** scored even where they got
some: a five-turn prefix of a sixteen-turn script is a different
stimulus, and folding it into a band built from whole
conversations compares two different things. The invocation still
raises when nothing at all finished, because a report over no runs
would pass every budget and say nothing.

Recovery sits at the conversation and not at the turn: the engine
already retries the provider call and counts what that cost in
`LLMMetrics.retries`, so a turn that reaches the runner has
exhausted that mechanism. What was actually being lost was the
other nineteen conversations.

**A run stops when the provider stops answering.** Once two
conversations in a row are lost to a timeout or a dropped
connection, no further conversation starts, and the ones in flight
end before their next turn. A turn already sent is never cut off.
The rest appear in `report.lost` as not played, and
`report.stopped_early` says the run stopped. A finished
conversation, or one lost to any other error, resets the count.
Pass `stop_after_unavailable=` to change the number, or `None` to
play every conversation regardless.

To avoid starting a run while the provider is down at all, ask it
first. `check_available()` sends one small request with a hard time
limit, straight to the provider, with no retry, no cache and no
fallback:

```python
check = agent.engine.check_available(timeout=15)
if not check.answered:
    print(f"Not answering: {check.error}")
```

```
Style contract: 42 replies over 3 runs, 0 LLM calls
  BROKEN one_message_when_dry            46% (6/13, <=0%)  runs 42%-100% !
  BROKEN opens_in_lower_case             45% (18/40, <=25%)  runs 19%-73% !
    ok   ends_on_a_question              55% (23/42, measure)  runs 50%-64%
  !  = runs disagreed by more than half. The pooled rate is not a baseline.
```

**Every row carries the range across runs, not only the pooled
rate**, because a pooled rate cannot distinguish one bad run from
several middling ones, and those call for opposite responses. A
rule whose runs disagree by more than half is marked `!` and
listed by `report.volatile`: its pooled figure is an average over
runs that behaved differently, so it predicts nothing about the
next one and must not be quoted as a baseline. This is not
hypothetical. The first contract written with this module scored a
rule at 81% on one repeat and 23% on three, and the pooled report
could not say which. `RuleResult.per_run` holds each run's score,
labelled `script#repeat`; a run the rule never applied to is left
out of the range rather than counted as zero, which would make a
rule look steadier than the runs that actually tested it.

**A rate divides by the turns the rule applied to, not by every
reply.** `when_prompt_words_at_most` above means the rule only
looks at turns where the counterpart was terse; without it the rate
would be diluted by every turn the rule was never about.
`forbidden_pattern` takes `when_prompt_matches` for the same
reason: a regular expression the counterpart's line must match
for the rule to apply, for a rule about how the agent answers one
kind of question. A rule
that no reply triggered is reported as `??` rather than passing
silently. A script that never reaches a rule is a gap in the
script, not evidence about the agent.

Built-in checks, all declarable by name in YAML: `forbidden_text`,
`forbidden_pattern`, `max_messages`, `max_line_chars`,
`ends_with_question`, `repeats_question_ending`,
`first_char_lowercase`, `capitalises_after_stop`, `max_emoji`,
`greets_after_opening`, `repeats_opener`,
`reproduces_verbatim`, `copies_example`. A rule may also be built in code with any
callable. It takes a `ReplyContext` and returns True for a
violation, False for a clean turn, and None when the rule does not
apply.

**`repeats_opener` compares conversations with each other.** It
takes no arguments and scores only each conversation's first reply:
a violation when another conversation in the same invocation opened
with the same line (ignoring casing and end punctuation). The rate
is the share of conversations whose opening was not their own.
A chat agent opened five of five with the same line while told
to vary it. Because it needs the whole invocation, the runner binds it
before scoring; calling it on one `ReplyContext` raises. A check of
your own can do the same by giving its callable a `bind_runs(runs)`
attribute that returns the per-reply check. The rate grows with the
number of conversations, so compare it only between invocations of
the same size.

**`copies_example` measures copying, not just identical lines.**
`reproduces_verbatim` only fires on a message equal to an example,
so a copy with one word swapped or a sentence appended passes. In
a chat agent's 13th September 2026 read that was most of them.
`copies_example(lines, min_similarity=0.7, min_chars=16)` counts a
message that contains an example line, or that scores at least
`min_similarity` against one (`difflib`). At 0.7 that read split
cleanly: every copy scored 0.71 or above, and the closest line the
agent wrote itself scored 0.58. Lines and messages under
`min_chars` are skipped, because a short stock reply ("recibido")
is not a copied example.

**`first_char_lowercase` and `capitalises_after_stop` are two
rules, and a persona usually insists on only the second.** Opening
a chat message in lower case is a register choice. Capitalising a
new sentence *inside* the message is not. One chat persona asks
for both in one paragraph and calls only the second non-optional.
Its contract read the insistence as attaching to the first, so
for weeks it enforced a preference at `budget: 0.25` while the
actual rule went unmeasured. When it was finally measured, the
first recorded read had broken it in 33% of the replies it applied
to, including the exact sentence the persona uses as its example of
what not to write.

To score conversations that already happened, such as a past read
or a real conversation, use `score_transcripts`, which makes no LLM
calls:

```python
from agents7.eval import score_transcripts

report = score_transcripts(contract, [
    [(None, "¡Buenas!"), ("bien", "¿bien o bien de no preguntes?")],
])
```

Pass one list per conversation rather than one flat list. Each
conversation is numbered from zero and its first reply has nothing
before it, so a second conversation's legitimate opening greeting
is not read as a greeting mid-chat, and its first reply is not read
as following the previous conversation's last.

The report carries every LLM call the run made, guard checks and
regenerations included, via the agent's `call_logger`. A contract
that has to be run repeatedly can therefore say what it cost.
`report.cost_cents` is `None` rather than `0.0` when no call was
priced, so an unpriced engine is never reported as a free one.

### Comparing runs over time

**One invocation is not a baseline.** Measured on a real agent:
three invocations of the *same* config, the *same* scripted
counterpart and the *same* code scored one rule at 23%, then 70%,
then 41%. Ten runs inside a single invocation agreed well enough to
look like a measurement; the invocations did not agree with each
other. So the unit worth persisting is the whole invocation:

```python
from agents7.eval import (
    append_style_run, compare_to_history,
    load_style_history, record_style_run,
)

record = record_style_run(report, contract, build_agent(), "v0.1.19")
history = load_style_history("data/style_history.jsonl")
print(compare_to_history(record, history).summary())
append_style_run(record, "data/style_history.jsonl")
```

```
Against 2 prior invocation(s) of the same configuration (4f8e4071c8dd):
  better opens_in_lower_case      8%  (was 12%-20%, margin 4%)
         cold 12%  warm 3%
    ==   one_message_when_dry    41%  (was 23%-70%, margin 29%)
    ==   ends_on_a_question      48%  (was 44%-55%, margin 7%)
         cold 44%  warm 53%
  Indented lines split a rate by script. Only the combined rate is judged.
  margin = how far this rate could sit from its own truth by
  chance. A rate outside its band by less than this is not a change.
```

The indented line appears when two or more scripts exercised a
rule and at least one reply broke it (since v9.4.0). Scripts are replayed in a fixed proportion,
so a move in one of them can hide inside the combined rate; the
split is there to be read, and the verdict stays on the combined
rate. `RuleRecord.by_script` and `RuleComparison.script_split` give
the same figures to code.

Six things make that trustworthy rather than merely tidy:

- **Configuration identity is computed, not claimed.**
  `fingerprint_run` hashes the agent's resolved system prompt with
  the contract's rule ids and script lines. A baseline is built
  only from records sharing it, so editing a persona starts a fresh
  baseline instead of being judged against a band that described a
  different agent. A version label would not do, because it is
  asserted by whoever types it, and a config can change without one
  moving.
  **A rule's budget is not in the hash** (since v8.0.0): it is read
  once, against a rate that has already been measured, so it cannot
  move a number. Keying on it meant that writing down the threshold
  a baseline had just justified discarded that baseline. The one
  edit guaranteed to change nothing also reset every band. Adding
  or removing a rule still changes the hash, because the ids do.
- **Each rule keeps its own history** (since v9.1.0). Comparison
  keys on `configuration_key`, the fingerprint without the rule
  ids, so adding a rule to a contract leaves every other rule's
  band in place. Each rule's band then holds only runs scored
  under the same `definition`: the check's name and arguments,
  which `build_rule` hashes onto `StyleRule` and
  `record_style_run` stores on `RuleRecord`. Changing what a rule
  checks under an unchanged id starts that rule afresh instead of
  judging new numbers against the old check's band. Records
  written before v9.1.0 carry neither field and compare exactly as
  they did.
- **The guard counts, and whether it ran is read from the run.** A
  guard rewrites replies before they are scored, so a run made
  without one measures something else: on a chat agent, dropping
  the guard moved one rule from 50% to 56%. When a guard took part, its resolved
  prompt and configured model go into the hash, and
  `StyleRunRecord`'s `guard_model_id` names that configured model.
  The guard is often a different model at a different price, and on
  that agent it is 59% of the bill. Since v9.3.0
  `guard_answered_by` records the model the provider says answered
  the guard, and a band only holds runs whose guard was answered by
  the same one, as below for the chatter. Participation cannot be read off the agent,
  because an unguarded run is made by replacing the guard's check
  while the agent keeps a fully configured `guard`; so
  `fingerprint_run` takes the report and reads the calls the run
  actually made. Pass it: `fingerprint_run(contract, agent,
  report)`. An agent with no guard hashes exactly as it did
  before, so unguarded history keeps its baseline.
- **A band only holds runs answered by the same model** (since
  v9.0.0). The chatter's model stays out of the hash, because it
  is the thing under test. Instead, `StyleRunRecord.answered_by`
  records the models the provider says answered (`CallRecord.model`),
  and `compare_to_history` leaves out runs of the same configuration
  answered by any other model. It lists them in `other_models` and
  names them in the summary, so a switch is still reported. This
  matters because a provider can move an alias under an unchanged
  config: DeepSeek moved `deepseek-chat` to `deepseek-flash` on
  10th September 2026, a chat agent's casing rule went from 75% to
  10%, and a band spanning both would pass almost anything. Runs
  that recorded no model are left out as soon as the current one
  records a model.
- **No verdict below two prior invocations.** One earlier run is a
  point, and a point has no spread. The comparison returns
  `Verdict.NO_BASELINE` and says so.
- **A change means outside the band *and* outside it by more than
  the rate's own sampling margin** (`sampling_margin`, twice the
  spread, floored at half an event). Without the margin,
  four stray events out of 175 flagged as a regression against a
  band built from one and one.
- **A rarely broken rule is shown, not judged** (since v9.6.0). When
  the prior invocations broke a rule fewer than
  `RARE_VIOLATIONS_PER_INVOCATION` (one) times each on average, its
  verdict is `Verdict.RARE`: the summary prints its rate marked
  `rare`, and it is never in `worse` or `better`. A band built mostly
  from zeros cannot tell a regression from bad luck: a chat agent's
  gate failed `no_trailing_laugh` twice at 3 of 111 replies on an
  unchanged config, with the laughs in different conversations each
  time. The price is that a quiet rule which jumps is not caught
  either. The same agent once scored `one_message_when_dry` at 10 of
  48 after runs of 1, 1, 0 and 0, so read the `rare` lines.
- **The margin is measured from the prior invocations themselves**,
  once there are four of them (`MIN_SPREAD_PRIORS`). What is being
  judged is whether the *invocation's* score moved, so the honest
  bar is how much that score has moved before: the prior rates are
  draws of exactly that quantity, with no independence assumption
  and no pooling of one script with another. Below four priors the
  margin falls back to the proxies below, and `margin_source` says
  which of the three it came from. The summary marks the two
  fallbacks with `*` and `**`, because they disagree with the real
  figure by two to three times.

  The priors' spread is read at the **upper end of its 90%
  confidence interval** (`SPREAD_CONFIDENCE`), not at its point
  value. Four values carry three degrees of freedom, so four that
  happen to agree used to give a margin of a point or so: a chat
  agent's gate failed an unchanged config at 18.4% against 14.9%-16.5%.
  Replayed over that gate's history, the bound took false gate
  failures on unchanged configs from six to none and the median
  margin from 9.2 to 16.7 points. The widening fades as priors
  accumulate.
- **Fallback one, the spread across this invocation's own runs.** A
  contract plays a script, so its replies are not drawn
  independently. Every reply after the first answers an agent that
  has read its own previous ones. The independent unit is the run.
  On a chat agent, `ends_on_a_question` varies barely at all between
  conversations while `one_message_when_dry` scored 42%, 82% and
  81% across three, roughly three times the variance a pooled
  denominator assumes: the agent settles into a rhythm early and
  keeps it. Treating its 16 dry turns as 16 independent draws is
  what made three invocations of one unchanged config read as two
  regressions. Each record keeps every run's label, so this spread
  is measured *within* each script and pooled across them rather
  than taken around the grand mean. Every invocation replays every
  script in the same proportion, so the difference between two of
  them is fixed structure rather than noise.
- **Fallback two, the binomial figure**, for a record written
  before per-run rates were persisted. It assumes the replies were
  drawn independently, which they are not, so it understates the
  uncertainty.

  **Why the fallbacks are fallbacks.** A chat agent has six gate
  invocations of one configuration, so the real
  invocation-to-invocation movement can be read off and compared
  against what the within-run proxy printed:
  `two_questions_in_a_row` 20% against a real 6%,
  `ends_on_a_question` 16% against 9%, `capitalises_after_stop` 18%
  against 11%, `one_message_when_dry` 37% against 19%. Those four
  rules were held to a bar two to three times too wide, where a
  real regression passes silently. And `opens_in_lower_case` printed 12%
  where the real movement is 20%, so it is not wrong in one
  direction only and cannot be corrected by moving a constant.
  Reading the margin off the priors puts all five within a point or
  two of the measured movement.

  **The proxy is dropped, not kept as a floor.** Taking whichever
  of the two figures is larger would leave every too-wide rule
  exactly as wide as it was, which is the failure this closes. The
  half-an-event floor stays, so four priors that happen to agree
  still cannot gate on a single event.
- **The history is append-only**, because it is the only thing that
  can say how much a figure moves when nothing changed.

Each record also keeps the budget it was scored under, since a
threshold tightened later would otherwise make every earlier
invocation look like a failure it never was. It keeps the model
id, because a provider
serving a different model explains a moved number as readily as a
config edit does, and `lost_runs`, because an invocation that
measured fewer runs than it planned carries more uncertainty than
its `repeats` implies.

### Sizing a contract

A contract that cannot see the move you care about is a contract
that reports nothing, expensively. `runs_for_shift` answers how
many runs that takes; `detectable_shift` answers the same question
from the other end. Both return a `ShiftEstimate` carrying the
move, the run count that goes with it, and where the spread behind
it came from:

```python
from agents7.eval import detectable_shift, runs_for_shift

rates = result.run_rates                    # one rate per run
runs_for_shift(0.10, rates).runs            # runs to see ten points
detectable_shift(rates, 10).rate            # what ten runs would see
```

**Which spread it uses is the whole accuracy of the answer**, and
both take the same three sources as `sampling_margin`, in the same
order. Earlier invocations of this configuration come first, the
labelled runs inside one invocation next, bare rates last:

```python
prior = [r.rule("ends_on_a_question") for r in history]

detectable_shift(
    planned_runs=20,
    prior_rates=[p.rate for p in prior],     # draws of the real thing
    prior_runs_each=len(prior[0].run_rates), # the size they were at
    prior_invocations=len(prior),
)   # -> ShiftEstimate(rate=0.142, runs=20, source=MarginSource.PRIORS)
```

`prior_runs_each` is required rather than optional: a spread
measured at ten runs an invocation means something different at
twenty, so without it the figure cannot be rescaled and is
refused. Passing `runs=` a sequence of `RunScore` instead of
`run_rates` keeps two scripts' different true rates apart, which
the bare rates cannot do. See `sampling_margin`.

Measured on a chat agent's five committed invocations, the weakest
source disagreed with the strongest by between 0.38 and 3.76
times on the same rules. It claimed a 2.9-point move was visible
on `no_paragraph_lines` where the real bar is 7.6, and demanded
16.5 points on `capitalises_after_stop` where 4.6 would do. Wrong
in both directions, so no constant corrects it. That is why the
source is returned rather than assumed.

The bar a rate has to clear is the top of the band plus the
margin, and both shrink with the square root of the *runs*. So
four times the runs sees half the move, and a longer script sees
almost nothing extra. Asked how to make a contract more sensitive,
the intuitive answer is to lengthen the scripts; it is the wrong
one.

Two things the result does not say. It is a coin flip, not a
guarantee: a move exactly that size clears the bar about half the
time, and catching it reliably takes roughly twice the runs. And
**more history raises the bar**. The maximum of five prior
invocations sits further above the truth than the maximum of two,
which is a property of a min/max band rather than a defect, so
`prior_invocations` is part of the calculation. It also answers a
question the function invites: adding invocations does not buy
sensitivity, so the runs inside one are the only lever.

Sizing a chat agent's contract this way split it in two. Eleven
of its fifteen rules reach ten points inside the 20 runs an
invocation already has, and `capitalises_after_stop`, the
tightest rule that is actually broken, needs only 5. The rule
measuring how the agent answers a monosyllable needs 161, because
its spread is the agent picking a rhythm per conversation and
holding it.
That is real behaviour rather than measurement error, so it is the
price of the question. It is worth knowing before reading a
26-point move as a regression.

### Comparing two configurations against each other

`compare_to_history` answers *"did this invocation leave its
band"*, which judges one configuration against its own past. It has no
answer for *"do these two configurations wobble by different
amounts"*, because a band is a property of one configuration.
`compare_arms` is that second instrument: two sets of invocations,
read against each other.

```python
from agents7.eval import compare_arms, load_style_history

guarded = load_style_history(path, fingerprint="30d996ff9270")
plain = load_style_history(other, fingerprint="526788c30730")
print(compare_arms(guarded, plain, "guarded", "unguarded").summary())
```

It reports three things, and keeping them apart is the point:

- **Level** is the mean rate each way. An apparatus that repairs
  faults shows up here, not in the spread.
- **Invocation spread** is how far whole-invocation scores sit from
  one another. This is what `compare_to_history` builds a band
  from, so it is what decides whether an apparatus change affects
  a band.
- **Conversation spread** is the wobble between individual
  conversations, with binomial noise removed. Two arms at
  different mean rates carry different amounts of that noise, so
  comparing raw run spreads would report a difference in *level*
  as a difference in wobble.

**Every p-value comes with the size of difference the test could
have seen**, given by `RuleArms.spread_threshold` per rule and
`ArmComparison.panel_threshold` for the panel. A null result is
unreadable without it: it is the difference between having ruled
something out and having been unable to see it. That is the same
discipline `ShiftEstimate.source` applies to a figure's
provenance, applied here to its power.

**Significance is by exact permutation of whole invocations**,
never of conversations. Two conversations inside one invocation
are not exchangeable with two inside another, for the same reason
`sampling_margin` takes the run rather than the reply as the
independent unit.

**Only rules that actually fire enter the panel.** A rule with no
violations either way has a spread of zero on both sides, which is
a ratio of exactly one carrying no information; a dozen of those
drag the median to a dead heat whatever the live rules did. The
first version of this comparison reported 1.00x at p=1.000 over
fifteen of a chat agent's rules, seven of which never fired.

`trim=True` (the default) cuts the longer arm to the shorter one's
size, keeping the newest, so two deliberately interleaved arms sit
in the same window of time and a change in what the provider is
serving cannot masquerade as the difference under test. What it
removes is reported as `drift_left` / `drift_right` rather than
discarded silently. Pass `trim=False` for a before/after, where
the arms are whatever sizes they are.

Run on a guarded chat agent (5th September 2026) it settled the question it
was built for: across the eight rules that fire, the guarded arm
moved more on five and less on three, a median spread ratio of
1.07x at p = 0.794, where a consistent difference above 1.65x
would have registered. The guard is not itself a source of
variance.

### Where inside a conversation a rule breaks

A contract scores a rule per invocation and per run, which cannot
say *when* in a conversation the rule breaks. `decompose_rhythm`
takes one 0/1 row per conversation, with one column per turn the
rule applied to, and splits the violations into three effects:

- **Position** is the rate at each turn, and first half against
  second. A rule that holds early and decays shows up here.
- **Conversation** is how far conversation means sit apart once the
  position drift is removed, against what independent coin flips
  would give. A ratio well above 1 means whole conversations run
  high or low.
- **Momentum** is whether a broken turn still predicts the next one
  once both of the above are removed, by shuffling turns *within*
  each conversation.

```python
from agents7.eval import (
    StyleContract,
    decompose_rhythm,
    run_style_contract,
    turn_matrix,
)

cold = StyleContract(rules=contract.rules, scripts=[script], repeats=10)
report = run_style_contract(factory, cold)
rule = next(r for r in contract.rules if r.id == "one_message_when_dry")
rows = turn_matrix(report.replies, rule)
print(decompose_rhythm(rows).summary())
```

**Play one script.** Two scripts are two stimuli with different
true rates, and mixing them puts that gap into the conversation
spread. **Save `rows`**: a re-run is a different set of
conversations, while re-analysing a saved matrix costs nothing and
gives the same figures for the same `seed`.

The summary ends with the uncontrolled figure, which is the rate
after a violation against the rate after a clean turn. It is marked
as not to be quoted alone, because most of that gap is position and
conversation rather than memory.

Run on a chat agent's `one_message_when_dry` (5th September 2026, two
runs of ten cold conversations) it found the rule clean on the
first dry turn in 20 of 20 conversations, the rate roughly doubling
by the second half, and conversation means 2.6x wider than
independence allows. That is a decay rather than the examples,
which pointed the next fix somewhere else.

## Feedback on a Chat Agent

A style contract says a rule stopped holding. A read says whether
the agent is good company. Neither produces the thing that actually
improves it: the operator's correction of a line, and the rule that
correction becomes. That has been written by hand, in prose, in two
shapes: a rewrite (*"this is what he should have said"*) and a
rule (*"he is too verbose and does not mirror the user"*). Neither
was machine-readable.

`eval.chat_feedback` is one format for both. An entry carries a
rewrite, a rule, or both, because a real correction is usually
both:

```yaml
read_id: 2026-08-28-first-read
agent: david
agent_version: 0.1.4
golden: [warm]          # accepted as a target to imitate in full
entries:
    - conversation: warm
      turn: 3
      verdict: wrong                   # right | weak | wrong
      he_said: best question anyone has asked me all year
      should_say: good question, give me a second
      why: overstates the investment three messages in
      rule: No overstating how interested he is early on.
      lands_in: characters/david.yml
      shipped: v0.1.5                  # empty means outstanding
```

Points worth knowing:

- **`turn` is optional.** Leaving it out makes the entry
  structural, about the whole conversation. *"He writes more the
  less the user gives him"* is the motivating case: the most useful kind
  of feedback, anchored to no single line.
- **`right` is not the absence of criticism.** It marks a line good
  enough to imitate, and such a line becomes a target in its own
  right.
- **`he_said` sits beside the rewrite** so a correction reads as a
  pair without opening the recording, and so a rewrite cannot drift
  from the line it was correcting.
- **`golden` names the conversations accepted whole.** The rest
  contribute corrections only. A correction taken from a
  conversation nobody accepted is still binding.

Three outputs, one source:

```python
feedback = load_feedback("docs/reads/<date>/feedback.yml")

feedback.rule_changes()   # grouped by config file, unshipped flagged
feedback.eval_rows()      # corrected turns as machine-readable targets
render_feedback(feedback) # the operator's report, as Markdown
```

Or from the CLI, which makes no LLM calls:

```bash
agents7 chat-feedback docs/reads/<date>/feedback.yml \
    --report docs/reads/<date>/feedback.md \
    --rows data/golden/<name>.jsonl
```

`merge_entries` replaces every entry on a turn the operator
touched and leaves every other turn alone, so an editing UI can
read a file, add to it and write it back without losing anything
hand-written.
