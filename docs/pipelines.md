# Pipelines

## Agent Pipelines

A `Pipeline` chains multiple agents sequentially. The output of each step is mapped into the input for the next step via an optional custom mapper function. Pipelines support single and batch modes, both sync and async.

Use cases:
- Multi-stage extraction (extract entities, then classify them)
- Enrichment chains (extract data, then augment with computed fields)
- Validation pipelines (extract, then verify with a second agent)

If any step produces `model=None` (parse failure or validator rejection) or a mapper raises, remaining steps are skipped and the pipeline returns `success=False`.

**Python configuration:**

```python
from pydantic import BaseModel
from agents7.agents import Labeler, Pipeline, PipelineStep
from agents7.models import AgentResponse

class StepOneOutput(BaseModel):
    name: str

class StepTwoOutput(BaseModel):
    greeting: str

def name_to_greeting(response: AgentResponse) -> dict:
    return {"text": f"Hello, {response.model.name}!"}

agent1 = Labeler(
    name="Extractor", version=1, engine=engine,
    system_prompt="Extract the person's name.",
    output_model=StepOneOutput,
)
agent2 = Labeler(
    name="Greeter", version=1, engine=engine,
    system_prompt="Generate a greeting.",
    output_model=StepTwoOutput,
)

pipeline = Pipeline(steps=[
    PipelineStep(agent=agent1, mapper=name_to_greeting),
    PipelineStep(agent=agent2),
])

# Single input
result = pipeline.act({"text": "Sarah from London"})
print(result.model.greeting)  # "Hello, Sarah!"
print(result.success)         # True

# Batch input
results = pipeline.act_in_batch([
    {"text": "Sarah from London"},
    {"text": "Dr. Chen from Singapore"},
])
for r in results:
    print(r.model.greeting if r.success else "Failed")
```

**YAML configuration:**

```yaml
steps:
  - agent: configs/extractor.yml
    mapper: mymodule.name_to_greeting
  - agent: configs/greeter.yml
    agent_type: agents7.agents.labeler.Labeler
```

- `agent`: Path to the agent's YAML config file (required)
- `agent_type`: Dotted import path to the agent class (optional, defaults to `Labeler`)
- `mapper`: Dotted import path to a mapper function (optional, defaults to `model.model_dump()`)
- `condition`: Dotted import path to a condition function (optional, defaults to always execute)

**Conditional steps:** Add a `condition` callable to `PipelineStep` to gate execution. The condition receives the current input dict and returns `bool`. If `False`, the step is skipped and the input passes unchanged to the next step:

```python
def has_name(input_data: dict) -> bool:
    return bool(input_data.get("name"))

pipeline = Pipeline(steps=[
    PipelineStep(agent=extractor),
    PipelineStep(agent=enricher, condition=has_name),
    PipelineStep(agent=formatter),
])
```

In batch mode, conditions are evaluated per-item: items that fail the condition skip the step while others proceed normally.

For simple sequential chains without custom mappers, use `from_agents`:

```python
pipeline = Pipeline.from_agents([agent1, agent2, agent3])
result = pipeline.act({"text": "input"})
```

**Pipeline streaming:** Stream the final step's output while running intermediate steps normally:

```python
stream = pipeline.act_stream({"text": "input"})
for chunk in stream:
    print(chunk, end="", flush=True)

print(stream.agent_response.model)

# Async version
stream = await pipeline.act_stream_async({"text": "input"})
async for chunk in stream:
    print(chunk, end="", flush=True)
```

If an intermediate step fails, the stream yields nothing and `agent_response.model` is `None`.
