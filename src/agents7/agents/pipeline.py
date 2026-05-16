import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from agents7.agents.base import BaseAgent
from agents7.agents.labeler import Labeler
from agents7.agents.streaming import (
    AgentAsyncStreamResult,
    AgentStreamResult,
)
from agents7.agents.utils import import_from_path, run_with_semaphore
from agents7.exceptions import PipelineConfigurationError
from agents7.logs import logger
from agents7.models import AgentResponse
from agents7.settings import BatchSettings


@dataclass(frozen=True)
class PipelineStep:
    """
    A single step in a Pipeline.

    Attributes:
        agent (BaseAgent): The agent to execute at this step.
        mapper (Callable[[AgentResponse], dict]): Function that
            maps the agent response to the input dict for the
            next step. Defaults to `default_mapper` which
            calls `model.model_dump()`.
        condition (Callable[[dict], bool] | None): Optional
            gate evaluated *before* executing the step. It
            receives the current input dict. If it returns
            ``False``, the step is skipped and the input is
            passed unchanged to the next step. Defaults to
            ``None`` (always execute).
    """

    agent: BaseAgent
    mapper: Callable[[AgentResponse], dict] | None = None
    condition: Callable[[dict], bool] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.agent, BaseAgent):
            raise PipelineConfigurationError(
                "`agent` must be a BaseAgent instance."
            )
        if self.mapper is None:
            object.__setattr__(self, "mapper", PipelineStep.default_mapper)
        if not callable(self.mapper):
            raise PipelineConfigurationError("`mapper` must be callable.")
        if self.condition is not None and not callable(self.condition):
            raise PipelineConfigurationError("`condition` must be callable.")

    @staticmethod
    def default_mapper(response: AgentResponse) -> dict:
        """Default mapper: converts the response model to a dict.

        :param response: The AgentResponse from the previous step.
        :return: Dictionary representation of the model.
        """
        return response.model.model_dump()


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

    @classmethod
    def from_agents(cls, agents: list[BaseAgent]) -> "Pipeline":
        """Create a Pipeline from a list of agents.

        Each agent is wrapped in a `PipelineStep` with the
        default mapper (`model.model_dump()`).

        :param agents: Ordered list of agents to chain.
        :return: A Pipeline instance.
        """
        return cls(steps=[PipelineStep(agent=a) for a in agents])

    def __post_init__(self) -> None:
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

    def _process_step_result(
        self,
        step_idx: int,
        step: PipelineStep,
        response: AgentResponse,
        executed_steps: list[AgentResponse],
    ) -> "PipelineResponse | dict | None":
        """Evaluate a single pipeline step result.

        :param step_idx: Index of the current step.
        :param step: The pipeline step that was executed.
        :param response: The agent response from the step.
        :param executed_steps: Accumulator for step
            responses (modified in place).
        :return: PipelineResponse on failure (stops the
            pipeline), a dict with the next input on
            intermediate success, or None on final-step
            success.
        """
        executed_steps.append(response)
        if response.model is None:
            return self._pipeline_result(executed_steps, False)
        if step_idx < len(self.steps) - 1:
            try:
                return step.mapper(response)
            except Exception as e:
                logger.error(
                    "Pipeline mapper failed at step %d: %s",
                    step_idx,
                    e,
                    exc_info=True,
                )
                return self._pipeline_result(executed_steps, False)
        return None

    @staticmethod
    def _pipeline_result(
        executed_steps: list[AgentResponse],
        success: bool,
    ) -> "PipelineResponse":
        """Build a PipelineResponse from executed steps."""
        return PipelineResponse(
            steps=executed_steps,
            model=(
                executed_steps[-1].model
                if success and executed_steps
                else None
            ),
            success=success,
        )

    def _evaluate_batch_step(
        self,
        step_idx: int,
        step: PipelineStep,
        ordered_indices: list[int],
        batch_responses: list[AgentResponse],
        per_input_steps: dict[int, list[AgentResponse]],
        failed: dict[int, None],
    ) -> dict[int, dict]:
        """Evaluate batch responses for one pipeline step.

        Updates ``per_input_steps`` and ``failed`` in place.

        :return: Active items for the next step.
        """
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
                        "Pipeline mapper failed at step %d for item %d: %s",
                        step_idx,
                        orig_idx,
                        e,
                        exc_info=True,
                    )
                    failed[orig_idx] = None
            else:
                next_active[orig_idx] = {}
        return next_active

    @staticmethod
    def _assemble_batch_results(
        n: int,
        per_input_steps: dict[int, list[AgentResponse]],
        failed: dict[int, None],
    ) -> list[PipelineResponse]:
        """Assemble batch results in original input order."""
        return [
            Pipeline._pipeline_result(
                per_input_steps[i],
                i not in failed and bool(per_input_steps[i]),
            )
            for i in range(n)
        ]

    def act(
        self,
        input_data: dict,
        chat_kwargs: dict | None = None,
    ) -> PipelineResponse:
        """
        Execute the pipeline synchronously on a single input.
        :param input_data: Input dict for the first step.
        :param chat_kwargs: Extra kwargs passed to each
            agent's ``act()`` method.
        :return: A PipelineResponse with all step results.
        """
        current_input = input_data
        executed_steps: list[AgentResponse] = []

        for step_idx, step in enumerate(self.steps):
            if step.condition is not None and not step.condition(
                current_input
            ):
                continue
            response = step.agent.act(current_input, chat_kwargs=chat_kwargs)
            result = self._process_step_result(
                step_idx, step, response, executed_steps
            )
            if isinstance(result, PipelineResponse):
                return result
            if result is not None:
                current_input = result

        return self._pipeline_result(executed_steps, True)

    async def act_async(
        self,
        input_data: dict,
        chat_kwargs: dict | None = None,
    ) -> PipelineResponse:
        """
        Execute the pipeline asynchronously on a single
        input.
        :param input_data: Input dict for the first step.
        :param chat_kwargs: Extra kwargs passed to each
            agent's ``act_async()`` method.
        :return: A PipelineResponse with all step results.
        """
        current_input = input_data
        executed_steps: list[AgentResponse] = []

        for step_idx, step in enumerate(self.steps):
            if step.condition is not None and not step.condition(
                current_input
            ):
                continue
            response = await step.agent.act_async(
                current_input, chat_kwargs=chat_kwargs
            )
            result = self._process_step_result(
                step_idx, step, response, executed_steps
            )
            if isinstance(result, PipelineResponse):
                return result
            if result is not None:
                current_input = result

        return self._pipeline_result(executed_steps, True)

    def act_stream(
        self,
        input_data: dict,
        chat_kwargs: dict | None = None,
    ) -> AgentStreamResult:
        """Execute the pipeline, streaming the final step.

        All steps except the last are executed with ``act()``.
        The last step uses ``act_stream()`` to yield text
        chunks as they arrive.

        If an intermediate step fails (``model=None`` or
        mapper error), a pre-filled ``AgentStreamResult``
        with ``success=False`` is returned.

        :param input_data: Input dict for the first step.
        :param chat_kwargs: Extra kwargs passed to each
            agent.
        :return: An `AgentStreamResult` for the final step.
        """
        current_input = input_data
        for step_idx, step in enumerate(self.steps[:-1]):
            if step.condition is not None and not step.condition(
                current_input
            ):
                continue
            response = step.agent.act(current_input, chat_kwargs=chat_kwargs)
            if response.model is None:
                return AgentStreamResult.from_failed(response)
            try:
                current_input = step.mapper(response)
            except Exception as e:
                logger.error(
                    "Pipeline mapper failed at step %d: %s",
                    step_idx,
                    e,
                    exc_info=True,
                )
                return AgentStreamResult.from_failed(response)

        last_step = self.steps[-1]
        if last_step.condition is not None and not last_step.condition(
            current_input
        ):
            return AgentStreamResult.from_failed(None)

        return last_step.agent.act_stream(
            current_input, chat_kwargs=chat_kwargs
        )

    async def act_stream_async(
        self,
        input_data: dict,
        chat_kwargs: dict | None = None,
    ) -> AgentAsyncStreamResult:
        """Execute the pipeline, streaming the final step
        asynchronously.

        All steps except the last are executed with
        ``act_async()``. The last step uses
        ``act_stream_async()``.

        :param input_data: Input dict for the first step.
        :param chat_kwargs: Extra kwargs passed to each
            agent.
        :return: An `AgentAsyncStreamResult` for the final
            step.
        """
        current_input = input_data
        for step_idx, step in enumerate(self.steps[:-1]):
            if step.condition is not None and not step.condition(
                current_input
            ):
                continue
            response = await step.agent.act_async(
                current_input, chat_kwargs=chat_kwargs
            )
            if response.model is None:
                return AgentAsyncStreamResult.from_failed(response)
            try:
                current_input = step.mapper(response)
            except Exception as e:
                logger.error(
                    "Pipeline mapper failed at step %d: %s",
                    step_idx,
                    e,
                    exc_info=True,
                )
                return AgentAsyncStreamResult.from_failed(response)

        last_step = self.steps[-1]
        if last_step.condition is not None and not last_step.condition(
            current_input
        ):
            return AgentAsyncStreamResult.from_failed(None)

        return await last_step.agent.act_stream_async(
            current_input, chat_kwargs=chat_kwargs
        )

    @staticmethod
    def _partition_by_condition(
        step: PipelineStep,
        active: dict[int, dict],
    ) -> tuple[list[int], dict[int, dict]]:
        """Split active items into those that pass the
        condition and those that are skipped.

        :param step: The pipeline step with an optional
            condition.
        :param active: Currently active items.
        :return: Tuple of (ordered indices to execute,
            skipped items dict).
        """
        if step.condition is None:
            return sorted(active.keys()), {}
        to_run: list[int] = []
        skipped: dict[int, dict] = {}
        for idx in sorted(active.keys()):
            if step.condition(active[idx]):
                to_run.append(idx)
            else:
                skipped[idx] = active[idx]
        return to_run, skipped

    def act_in_batch(
        self,
        input_list: list[dict],
        chat_kwargs: dict | None = None,
    ) -> list[PipelineResponse]:
        """
        Execute the pipeline on a batch of inputs. Each step
        processes all surviving items. Failed items are
        excluded from subsequent steps.
        :param input_list: List of input dicts.
        :param chat_kwargs: Extra kwargs passed to each
            agent's ``act()`` method.
        :return: List of PipelineResponse, one per input,
            in original order.
        """
        n = len(input_list)
        active: dict[int, dict] = {i: inp for i, inp in enumerate(input_list)}
        per_input_steps: dict[int, list[AgentResponse]] = {
            i: [] for i in range(n)
        }
        failed: dict[int, None] = {}

        for step_idx, step in enumerate(self.steps):
            if not active:
                break
            ordered, skipped = self._partition_by_condition(step, active)
            if not ordered:
                active = skipped
                continue
            batch_inputs = [active[i] for i in ordered]
            batch_responses = [
                step.agent.act(inp, chat_kwargs=chat_kwargs)
                for inp in batch_inputs
            ]
            active = self._evaluate_batch_step(
                step_idx,
                step,
                ordered,
                batch_responses,
                per_input_steps,
                failed,
            )
            active.update(skipped)

        return self._assemble_batch_results(n, per_input_steps, failed)

    async def act_in_batch_async(
        self,
        input_list: list[dict],
        chat_kwargs: dict | None = None,
        batch_settings: BatchSettings | None = None,
    ) -> list[PipelineResponse]:
        """
        Execute the pipeline asynchronously on a batch of
        inputs. Each step processes all surviving items
        concurrently. Failed items are excluded from
        subsequent steps.
        :param input_list: List of input dicts.
        :param chat_kwargs: Extra kwargs passed to each
            agent's ``act_async()`` method.
        :param batch_settings: Optional `BatchSettings`
            instance controlling concurrency. Defaults to
            `None` which uses `BatchSettings()`. Fields:
            - max_concurrent (int | None): Maximum number
              of concurrent LLM requests. Overridable via
              the `BATCH_MAX_CONCURRENT` env var. Defaults
              to 5. `None` means unlimited concurrency.
        :return: List of PipelineResponse, one per input,
            in original order.
        """
        settings = batch_settings or BatchSettings()
        sem = (
            asyncio.Semaphore(settings.max_concurrent)
            if settings.max_concurrent
            else None
        )

        n = len(input_list)
        active: dict[int, dict] = {i: inp for i, inp in enumerate(input_list)}
        per_input_steps: dict[int, list[AgentResponse]] = {
            i: [] for i in range(n)
        }
        failed: dict[int, None] = {}

        for step_idx, step in enumerate(self.steps):
            if not active:
                break
            ordered, skipped = self._partition_by_condition(step, active)
            if not ordered:
                active = skipped
                continue
            batch_inputs = [active[i] for i in ordered]
            batch_responses = await asyncio.gather(
                *[
                    run_with_semaphore(
                        step.agent.act_async(
                            inp,
                            chat_kwargs=chat_kwargs,
                        ),
                        sem,
                    )
                    for inp in batch_inputs
                ]
            )
            active = self._evaluate_batch_step(
                step_idx,
                step,
                ordered,
                batch_responses,
                per_input_steps,
                failed,
            )
            active.update(skipped)

        return self._assemble_batch_results(n, per_input_steps, failed)

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
        import yaml

        with open(file_path, encoding="utf-8") as _f:
            config = yaml.safe_load(_f) or {}

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
                agent_cls = Labeler

            agent = agent_cls.from_yaml(step_config["agent"])

            # Resolve mapper
            mapper_path = step_config.get("mapper")
            if mapper_path:
                mapper = import_from_path(mapper_path)
            else:
                mapper = PipelineStep.default_mapper

            # Resolve condition
            condition_path = step_config.get("condition")
            condition = (
                import_from_path(condition_path) if condition_path else None
            )

            pipeline_steps.append(
                PipelineStep(
                    agent=agent,
                    mapper=mapper,
                    condition=condition,
                )
            )

        return cls(steps=pipeline_steps)
