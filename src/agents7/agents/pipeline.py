import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

from cpinto.files import load_yaml

from agents7.agents.base import BaseAgent
from agents7.agents.utils import import_from_path
from agents7.exceptions import PipelineConfigurationError
from agents7.logs import logger
from agents7.models import AgentResponse
from agents7.settings import BatchSettings


def _default_mapper(response: AgentResponse) -> dict:
    """
    Default mapper: converts the response model to a dict.
    :param response: The AgentResponse from the previous step.
    :return: Dictionary representation of the model.
    """
    return response.model.model_dump()


@dataclass(frozen=True)
class PipelineStep:
    """
    A single step in a Pipeline.

    Attributes:
        agent (BaseAgent): The agent to execute at this step.
        mapper (Callable[[AgentResponse], dict]): Function that
            maps the agent response to the input dict for the
            next step. Defaults to ``_default_mapper`` which
            calls ``model.model_dump()``.
    """

    agent: BaseAgent
    mapper: Callable[[AgentResponse], dict] = field(default=_default_mapper)

    def __post_init__(self):
        if not isinstance(self.agent, BaseAgent):
            raise PipelineConfigurationError(
                "`agent` must be a BaseAgent instance."
            )
        if not callable(self.mapper):
            raise PipelineConfigurationError("`mapper` must be callable.")


class PipelineResponse(BaseModel):
    """
    Response from a full pipeline execution.

    Attributes:
        steps (list[AgentResponse]): One AgentResponse per
            executed pipeline step.
        model (BaseModel | None): The final model produced by
            the last step, or None if the pipeline failed.
        success (bool): True if all steps completed
            successfully.
    """

    steps: list[AgentResponse]
    model: BaseModel | None
    success: bool


@dataclass(frozen=True)
class Pipeline:
    """
    Chains multiple agents sequentially. Output of each step
    is mapped (via an optional custom function) into the input
    for the next step.

    Attributes:
        steps (list[PipelineStep]): Ordered list of pipeline
            steps to execute.
    """

    steps: list[PipelineStep]

    def __post_init__(self):
        if not self.steps:
            raise PipelineConfigurationError(
                "`steps` must be a non-empty list."
            )
        for i, step in enumerate(self.steps):
            if not isinstance(step, PipelineStep):
                raise PipelineConfigurationError(
                    f"Step {i} must be a PipelineStep instance, "
                    f"got {type(step).__name__}."
                )

    def act(
        self,
        input_data: dict,
        chat_kwargs: dict = None,
    ) -> PipelineResponse:
        """
        Execute the pipeline synchronously on a single input.
        :param input_data: Input dict for the first step.
        :param chat_kwargs: Extra kwargs passed to each agent's
            ``act()`` method.
        :return: A PipelineResponse with all step results.
        """
        current_input = input_data
        executed_steps: list[AgentResponse] = []

        for i, step in enumerate(self.steps):
            response = step.agent.act(current_input, chat_kwargs=chat_kwargs)
            executed_steps.append(response)

            if response.model is None:
                return PipelineResponse(
                    steps=executed_steps,
                    model=None,
                    success=False,
                )

            # Skip mapper on last step
            if i < len(self.steps) - 1:
                try:
                    current_input = step.mapper(response)
                except Exception as e:
                    logger.error(f"Pipeline mapper failed at step {i}: {e}")
                    return PipelineResponse(
                        steps=executed_steps,
                        model=None,
                        success=False,
                    )

        return PipelineResponse(
            steps=executed_steps,
            model=executed_steps[-1].model,
            success=True,
        )

    async def act_async(
        self,
        input_data: dict,
        chat_kwargs: dict = None,
    ) -> PipelineResponse:
        """
        Execute the pipeline asynchronously on a single input.
        :param input_data: Input dict for the first step.
        :param chat_kwargs: Extra kwargs passed to each agent's
            ``act_async()`` method.
        :return: A PipelineResponse with all step results.
        """
        current_input = input_data
        executed_steps: list[AgentResponse] = []

        for i, step in enumerate(self.steps):
            response = await step.agent.act_async(
                current_input, chat_kwargs=chat_kwargs
            )
            executed_steps.append(response)

            if response.model is None:
                return PipelineResponse(
                    steps=executed_steps,
                    model=None,
                    success=False,
                )

            # Skip mapper on last step
            if i < len(self.steps) - 1:
                try:
                    current_input = step.mapper(response)
                except Exception as e:
                    logger.error(f"Pipeline mapper failed at step {i}: {e}")
                    return PipelineResponse(
                        steps=executed_steps,
                        model=None,
                        success=False,
                    )

        return PipelineResponse(
            steps=executed_steps,
            model=executed_steps[-1].model,
            success=True,
        )

    def act_in_batch(
        self,
        input_list: list[dict],
        chat_kwargs: dict = None,
    ) -> list[PipelineResponse]:
        """
        Execute the pipeline on a batch of inputs. Each step
        processes all surviving items in one batch call. Failed
        items are excluded from subsequent steps.
        :param input_list: List of input dicts.
        :param chat_kwargs: Extra kwargs passed to each agent's
            ``act()`` method.
        :return: List of PipelineResponse, one per input, in
            original order.
        """
        n = len(input_list)
        # Track active items: orig_index -> current_input
        active: dict[int, dict] = {i: inp for i, inp in enumerate(input_list)}
        # Track per-input step results
        per_input_steps: dict[int, list[AgentResponse]] = {
            i: [] for i in range(n)
        }
        # Track failed items
        failed: dict[int, BaseModel | None] = {}

        for step_idx, step in enumerate(self.steps):
            if not active:
                break

            # Build ordered batch from active items
            ordered_indices = sorted(active.keys())
            batch_inputs = [active[i] for i in ordered_indices]

            # Process batch: one act() call per input
            batch_responses = [
                step.agent.act(inp, chat_kwargs=chat_kwargs)
                for inp in batch_inputs
            ]

            # Evaluate results
            next_active: dict[int, dict] = {}
            for orig_idx, response in zip(ordered_indices, batch_responses):
                per_input_steps[orig_idx].append(response)

                if response.model is None:
                    failed[orig_idx] = None
                    continue

                # Apply mapper (skip on last step)
                if step_idx < len(self.steps) - 1:
                    try:
                        next_active[orig_idx] = step.mapper(response)
                    except Exception as e:
                        logger.error(
                            f"Pipeline mapper failed at step "
                            f"{step_idx} for item {orig_idx}: {e}"
                        )
                        failed[orig_idx] = None
                else:
                    next_active[orig_idx] = {}  # placeholder

            active = next_active

        # Assemble results in original order
        results: list[PipelineResponse] = []
        for i in range(n):
            if i in failed:
                results.append(
                    PipelineResponse(
                        steps=per_input_steps[i],
                        model=None,
                        success=False,
                    )
                )
            else:
                steps = per_input_steps[i]
                results.append(
                    PipelineResponse(
                        steps=steps,
                        model=steps[-1].model if steps else None,
                        success=bool(steps),
                    )
                )
        return results

    async def act_in_batch_async(
        self,
        input_list: list[dict],
        chat_kwargs: dict = None,
        batch_settings: BatchSettings | None = None,
    ) -> list[PipelineResponse]:
        """
        Execute the pipeline asynchronously on a batch of
        inputs. Each step processes all surviving items
        concurrently. Failed items are excluded from
        subsequent steps.
        :param input_list: List of input dicts.
        :param chat_kwargs: Extra kwargs passed to each agent's
            ``act_async()`` method.
        :param batch_settings: Optional ``BatchSettings``
            instance controlling concurrency. Defaults to
            ``None`` which uses ``BatchSettings()``
            (max_concurrent=5).
        :return: List of PipelineResponse, one per input, in
            original order.
        """
        settings = batch_settings or BatchSettings()
        sem = (
            asyncio.Semaphore(settings.max_concurrent)
            if settings.max_concurrent
            else None
        )

        async def _limited(coro):
            if sem:
                async with sem:
                    return await coro
            return await coro

        n = len(input_list)
        active: dict[int, dict] = {i: inp for i, inp in enumerate(input_list)}
        per_input_steps: dict[int, list[AgentResponse]] = {
            i: [] for i in range(n)
        }
        failed: dict[int, BaseModel | None] = {}

        for step_idx, step in enumerate(self.steps):
            if not active:
                break

            ordered_indices = sorted(active.keys())
            batch_inputs = [active[i] for i in ordered_indices]

            batch_responses = list(
                await asyncio.gather(
                    *[
                        _limited(
                            step.agent.act_async(inp, chat_kwargs=chat_kwargs)
                        )
                        for inp in batch_inputs
                    ]
                )
            )

            next_active: dict[int, dict] = {}
            for orig_idx, response in zip(ordered_indices, batch_responses):
                per_input_steps[orig_idx].append(response)

                if response.model is None:
                    failed[orig_idx] = None
                    continue

                if step_idx < len(self.steps) - 1:
                    try:
                        next_active[orig_idx] = step.mapper(response)
                    except Exception as e:
                        logger.error(
                            f"Pipeline mapper failed at step "
                            f"{step_idx} for item {orig_idx}: "
                            f"{e}"
                        )
                        failed[orig_idx] = None
                else:
                    next_active[orig_idx] = {}

            active = next_active

        results: list[PipelineResponse] = []
        for i in range(n):
            if i in failed:
                results.append(
                    PipelineResponse(
                        steps=per_input_steps[i],
                        model=None,
                        success=False,
                    )
                )
            else:
                steps = per_input_steps[i]
                results.append(
                    PipelineResponse(
                        steps=steps,
                        model=steps[-1].model if steps else None,
                        success=bool(steps),
                    )
                )
        return results

    @classmethod
    def from_yaml(cls, file_path: str) -> "Pipeline":
        """
        Create a Pipeline from a YAML configuration file.
        :param file_path: Path to the YAML file. Schema:
            ```yaml
            steps:
              - agent: path/to/agent_config.yml
                agent_type: agents7.agents.Classifier  # optional
                mapper: mymodule.my_mapper             # optional
              - agent: path/to/agent2.yml
            ```
        :return: A Pipeline instance.
        """
        config = load_yaml(file_path)

        if "steps" not in config or not config["steps"]:
            raise PipelineConfigurationError(
                f"Missing or empty `steps` in pipeline YAML `{file_path}`."
            )

        pipeline_steps: list[PipelineStep] = []
        for i, step_config in enumerate(config["steps"]):
            if "agent" not in step_config:
                raise PipelineConfigurationError(
                    f"Step {i} missing `agent` key in `{file_path}`."
                )

            # Resolve agent class
            agent_type_path = step_config.get("agent_type")
            if agent_type_path:
                agent_cls = import_from_path(agent_type_path)
            else:
                from agents7.agents.labeler import Labeler

                agent_cls = Labeler

            agent = agent_cls.from_yaml(step_config["agent"])

            # Resolve mapper
            mapper_path = step_config.get("mapper")
            if mapper_path:
                mapper = import_from_path(mapper_path)
            else:
                mapper = _default_mapper

            pipeline_steps.append(PipelineStep(agent=agent, mapper=mapper))

        return cls(steps=pipeline_steps)
