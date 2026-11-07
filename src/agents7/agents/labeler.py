import json
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

from pydantic import BaseModel, create_model
from tqdm import tqdm

from agents7.agents.base import BaseAgent
from agents7.agents.batch import BatchProcessor
from agents7.agents.config_loader import load_yaml_settings
from agents7.agents.parsing import ResponseParser
from agents7.agents.retry import CallAttempt, ParseRetryHandler
from agents7.agents.streaming import (
    AgentAsyncStreamResult,
    AgentStreamResult,
)
from agents7.agents.utils import (
    get_examples_str,
    model_class_to_string,
)
from agents7.call_log import CallRecord, generate_trace_id, now
from agents7.exceptions import AgentConfigurationError, ParseError
from agents7.logs import logger
from agents7.models import AgentResponse
from agents7.models.base import LLMMetrics
from agents7.settings import BatchSettings


@dataclass(frozen=True)
class Labeler(BaseAgent):
    """
    Labeler agent for processing and labeling data.
    This agent is designed to analyze and label data based on
    predefined rules.

    Attributes:
        name (str): The name of the agent.
        version (int): The version of the agent.
        engine (LLMEngine): The LLM engine used by the agent.
        system_prompt (str): The system prompt for the agent.
        output_model (BaseModel): The model used to validate and
            structure the output.
        examples (list[dict], optional): List of examples to include in the
            system prompt. Each example should be a dictionary with 'input'
            and 'output' keys. Defaults to None.
        validators (list[Callable], optional): List of callables that
            run after the LLM response is parsed. Each receives a
            BaseModel and returns it (possibly modified). Raising
            rejects the model (sets it to None). Defaults to None.
        think (bool, optional): Whether to enable thinking mode
            for LLM calls. Defaults to False. Can be overridden
            per call via `chat_kwargs`.
        max_parse_retries (int, optional): Number of times to
            re-prompt the LLM when response parsing fails.
            On each retry the parse error is fed back to the
            model so it can correct its output. Defaults to 0
            (no retries). Only applies to ``act()`` and
            ``act_async()``; batch methods do not retry.
        call_logger (Callable[[CallRecord], None], optional):
            Callable invoked with a `CallRecord` for every LLM
            call made, containing trace ID, agent name, model,
            tokens, duration, cost, and success/failure. One
            call is the ordinary case; an ``act()`` or
            ``act_async()`` that needed parse retries emits one
            record per attempt, all sharing the answer's trace
            ID, so the log counts the calls that were made. A
            streamed call emits its record once the caller
            finishes iterating, since that is when the outcome
            is known. Defaults to None (no logging). Use
            `JsonCallLogger` for JSON output.
        run_id (str, optional): Name of the run these calls
            belong to, written to every `CallRecord` so a log
            can be grouped by run as well as by turn — an eval
            sweep, a benchmark, a batch job. Defaults to None,
            meaning the caller named no run.
        preprocessor (Callable, optional): Callable that
            transforms input data before it is sent to the
            LLM. Receives the raw input (dict or list) and
            must return the transformed input. Applied in
            ``act()``, ``act_async()``, ``act_stream()``,
            ``act_stream_async()``, and batch methods.
            Defaults to None (no preprocessing).
    """

    output_model: type[BaseModel]
    examples: list[dict] | None = None
    validators: list[Callable[[BaseModel], BaseModel]] | None = None
    think: bool = field(default=False, kw_only=True)
    max_parse_retries: int = field(default=0, kw_only=True)
    call_logger: Callable | None = field(default=None, kw_only=True)
    run_id: str | None = field(default=None, kw_only=True)
    preprocessor: Callable | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_output_model()
        self._initialize_parser()
        self._initialize_handlers()

    def _validate_output_model(self) -> None:
        if not issubclass(self.output_model, BaseModel):
            raise AgentConfigurationError(
                "`output_model` must be a subclass of `pydantic.BaseModel`"
            )

    def _initialize_parser(self) -> None:
        output_model_list = create_model(
            f"{self.output_model.__name__}List",
            items=(list[self.output_model], ...),
            __base__=BaseModel,
        )
        output_schema = self.output_model.model_json_schema()
        output_list_schema = output_model_list.model_json_schema()
        parser = ResponseParser(
            output_model=self.output_model,
            output_model_list=output_model_list,
            name=self.name,
            validators=self.validators,
        )
        object.__setattr__(self, "_output_model_list", output_model_list)
        object.__setattr__(self, "_output_schema", output_schema)
        object.__setattr__(self, "_output_list_schema", output_list_schema)
        object.__setattr__(self, "_parser", parser)

    def _initialize_handlers(self) -> None:
        object.__setattr__(
            self,
            "_retry_handler",
            ParseRetryHandler(
                engine=self.engine,
                parser=self._parser,
                name=self.name,
                max_retries=self.max_parse_retries,
                output_schema=self._output_schema,
            ),
        )
        object.__setattr__(
            self,
            "_batch_processor",
            BatchProcessor(
                engine=self.engine,
                parser=self._parser,
                name=self.name,
                preprocessor=self.preprocessor,
                output_list_schema=self._output_list_schema,
                on_call=self._emit_call_log,
            ),
        )

    @classmethod
    def from_yaml(cls, file_path: str) -> "Labeler":
        """
        Create an instance of the agent from a YAML file configuration.
        This method should be overridden by subclasses to provide specific
        YAML loading logic.

        :param file_path: The path to the YAML file containing the agent
            configuration. Schema:
            ```yaml
            name: <name of the agent>
            version: <version of the agent>
            engine:
                provider: <llm provider>
                model: <llm model>
            system_prompt: <system prompt for the agent>
            prompt_params:
                <key>: <value to substitute $key in system_prompt>
                ...
            system_examples:
                - input:
                    <example 1 input>
                  output:
                    <example 1 output>
                ...
            output_model: <path to the pydantic model class, e.g.
                "agents7.models.MyOutputModel">
            ```
            The fields `prompt_params`, `system_examples`, and
            `think` are optional.
        :return: An instance of the agent.
        """
        config = load_yaml_settings(file_path)
        return Labeler(**config)

    @property
    def prompt(self) -> str:
        """Full prompt used by the agent."""
        return self._build_prompt()

    @property
    def prompt_for_batch(self) -> str:
        """Full prompt for batch processing."""
        return self._build_prompt(
            "You will receive multiple inputs. Process "
            "each one independently — apply the exact same "
            "analysis to each item as you would for a "
            "single input. Do not default to a safe or "
            "neutral answer just because there are multiple "
            "items. Return an array of outputs in the same "
            "order as the inputs."
        )

    def _build_prompt(
        self,
        instruction: str = "",
        extra_sections: list[str] | None = None,
    ) -> str:
        """Build the full prompt from system prompt, schema,
        instruction, extra sections, and examples.

        Subclasses pass their agent-specific instruction
        and optional extra sections (e.g., constraints).

        :param instruction: Agent-specific instruction
            appended after the output model schema.
        :param extra_sections: Additional text sections
            inserted between instruction and examples.
        :return: Full prompt string.
        """
        full_prompt = (
            f"{self.resolved_system_prompt}\n\n"
            f"## Output Model Schema\n\n"
            f"{model_class_to_string(self.output_model)}"
        )
        if instruction:
            full_prompt = f"{full_prompt}\n\n{instruction}"
        full_prompt = full_prompt.strip()

        for section in extra_sections or []:
            if section:
                full_prompt = f"{full_prompt}\n\n{section}"

        if self.examples:
            full_prompt = (
                f"{full_prompt}\n\n{get_examples_str(self.examples)}"
            )

        return full_prompt

    @property
    def output_model_list(self) -> type[BaseModel]:
        """Dynamic model wrapping ``output_model`` in a list."""
        return self._output_model_list

    @property
    def output_schema(self) -> dict:
        """JSON schema dict for the output model."""
        return self._output_schema

    def enum_values(self) -> dict[str, dict[str, str]]:
        """Map of StrEnum fields to their ``{name: value}``
        members.

        :return: Dict mapping field names to enum member dicts.
            Empty dict if no StrEnum fields exist.
        """
        return self._parser._get_enum_mappings()

    def describe(self) -> str:
        """Human-readable summary of the agent.

        :return: Multi-line string describing the agent's
            name, version, model, output schema, and enum
            values.
        """
        lines = [
            f"Agent: {self.name} (v{self.version})",
            f"Model: {self.engine.model_id}",
            f"Output: {self.output_model.__name__}",
        ]
        enums = self.enum_values()
        if enums:
            lines.append("Enum fields:")
            for field_name, members in enums.items():
                vals = ", ".join(members.keys())
                lines.append(f"  {field_name}: {vals}")
        return "\n".join(lines)

    def act(
        self, input_data: dict | list[dict], chat_kwargs: dict = None
    ) -> AgentResponse:
        """
        Perform the labeling action based on the provided input.
        :param input_data: The input data to be processed.
        :param chat_kwargs: Dictionary with extra keyword arguments to
            send to the `self.engine.chat` method. Note that:
            - The parameter `temperature` is set to 0 for agents, and
              won't be overridden by this parameter.
            - The parameter `think` is set to True by default, unless
              specified otherwise.
            - Available parameters:
                - max_tokens: The maximum number of tokens to generate in the
                    response.
                - think: If True, the model will think before generating the
                    response.
                - seed: Seed for random number generation. Default to None.
        :return: An AgentResponse model, containing:
            - response: The generated text response.
            - model: The structured output validated by the output model.
            - thinking: The thinking part of the response, if any.
            - metrics: LLMMetrics model, containing:
                - tokens: The number of tokens in the response.
                - prompt_tokens: The number of tokens in the prompt.
                - total_duration: The total duration of the response,
                 in seconds.
            - idx: Optional index for tracking purposes.
        """
        messages, chat_kwargs = self._prepare_input(input_data, chat_kwargs)
        response = self._strip_role(
            self.engine.chat(
                messages=messages,
                temperature=0,
                output_format=self._output_schema,
                **chat_kwargs,
            )
        )
        attempts = [self._first_attempt(response, messages)]
        try:
            parsed = self._parser.parse(response["response"])
        except ParseError as err:
            parsed = self._retry_handler.retry_sync(
                err, messages, chat_kwargs, attempts
            )
        return self._finalize_response(response, parsed, attempts=attempts)

    async def act_async(
        self,
        input_data: dict | list[dict],
        chat_kwargs: dict | None = None,
        _idx: int = 0,
    ) -> AgentResponse:
        """Async version of `act()`.

        :param input_data: The input data to be processed.
        :param chat_kwargs: See `act()` for details.
        :param _idx: Internal index for batch tracking.
            Should not be set by the user.
        :return: See `act()` for return schema.
        """
        messages, chat_kwargs = self._prepare_input(input_data, chat_kwargs)
        response = self._strip_role(
            await self.engine.chat_async(
                messages=messages,
                temperature=0,
                output_format=self._output_schema,
                **chat_kwargs,
            )
        )
        attempts = [self._first_attempt(response, messages)]
        try:
            parsed = self._parser.parse(response["response"])
        except ParseError as err:
            parsed = await self._retry_handler.retry_async(
                err, messages, chat_kwargs, attempts
            )
        return self._finalize_response(
            response, parsed, idx=_idx, attempts=attempts
        )

    def act_stream(
        self,
        input_data: dict | list[dict],
        chat_kwargs: dict | None = None,
    ) -> AgentStreamResult:
        """Stream the labeling response, yielding text chunks.

        Returns an `AgentStreamResult` that yields text chunks
        when iterated. After iteration, access the parsed
        `AgentResponse` via `stream.agent_response`.

        A `CallRecord` is emitted once iteration completes, not
        when this method returns — the outcome is not known
        until the streamed text has been parsed. A caller that
        abandons the stream without iterating it records
        nothing.

        :param input_data: The input data to be processed.
        :param chat_kwargs: See `act()` for details.
        :return: An `AgentStreamResult` yielding text chunks.
        """
        messages, chat_kwargs = self._prepare_input(input_data, chat_kwargs)
        stream = self.engine.chat_stream(
            messages=messages,
            temperature=0,
            output_format=self._output_schema,
            **chat_kwargs,
        )
        return AgentStreamResult(
            stream,
            self._parser,
            on_complete=partial(self._emit_call_log, messages=messages),
        )

    async def act_stream_async(
        self,
        input_data: dict | list[dict],
        chat_kwargs: dict | None = None,
    ) -> AgentAsyncStreamResult:
        """Async version of `act_stream`.

        Returns an `AgentAsyncStreamResult` that yields text
        chunks when async-iterated. After iteration, access
        the parsed `AgentResponse` via
        `stream.agent_response`.

        As in `act_stream`, the `CallRecord` is emitted once
        iteration completes.

        :param input_data: The input data to be processed.
        :param chat_kwargs: See `act()` for details.
        :return: An `AgentAsyncStreamResult` yielding text
            chunks.
        """
        messages, chat_kwargs = self._prepare_input(input_data, chat_kwargs)
        stream = await self.engine.chat_stream_async(
            messages=messages,
            temperature=0,
            output_format=self._output_schema,
            **chat_kwargs,
        )
        return AgentAsyncStreamResult(
            stream,
            self._parser,
            on_complete=partial(self._emit_call_log, messages=messages),
        )

    def act_in_batch(
        self,
        input_list: list[dict | list[dict]],
        chat_kwargs: dict | None = None,
        requests_per_prompt: int = 10,
        iter_func: Callable = tqdm,
    ) -> list[AgentResponse]:
        """Sync batch processing.

        :param input_list: List of input data to process.
        :param chat_kwargs: See `act()` for details.
        :param requests_per_prompt: Items per LLM call.
            Defaults to 10.
        :param iter_func: Progress iterator. Defaults to
            `tqdm.tqdm`. Pass `None` to disable.
        :return: List of `AgentResponse`, one per input.
            See `act()` for the response schema.
        """
        return self._batch_processor.act_in_batch(
            input_list,
            self.prompt_for_batch,
            self._sanitize_chat_kwargs(chat_kwargs),
            requests_per_prompt=requests_per_prompt,
            iter_func=iter_func,
        )

    async def act_in_batch_async(
        self,
        input_list: list[dict | list[dict]],
        chat_kwargs: dict | None = None,
        requests_per_prompt: int = 10,
        with_tqdm: bool = True,
        batch_settings: BatchSettings | None = None,
    ) -> list[AgentResponse]:
        """Async batch processing.

        :param input_list: List of input data to process.
        :param chat_kwargs: See `act()` for details.
        :param requests_per_prompt: Items per LLM call.
            Defaults to 10.
        :param with_tqdm: Show async progress bar.
        :param batch_settings: Optional `BatchSettings`
            controlling concurrency. Defaults to
            `BatchSettings()` (max 5 concurrent).
        :return: List of `AgentResponse`, one per input.
            See `act()` for the response schema.
        """
        return await self._batch_processor.act_in_batch_async(
            input_list,
            self.prompt_for_batch,
            self._sanitize_chat_kwargs(chat_kwargs),
            requests_per_prompt=requests_per_prompt,
            with_tqdm=with_tqdm,
            batch_settings=batch_settings,
        )

    def _build_act_messages(
        self,
        input_data: dict | list[dict],
    ) -> list[dict[str, str]]:
        """Build the system + user message list for act().

        :param input_data: Preprocessed input data.
        :return: Messages list for engine.chat().
        """
        logger.debug(
            "Agent `%s`: processing input data:\n%s",
            self.name,
            input_data,
        )
        return [
            {"role": "system", "content": self.prompt},
            {
                "role": "user",
                "content": (f"Process the following:\n{input_data}"),
            },
        ]

    @staticmethod
    def _first_attempt(
        response: dict,
        messages: list[dict[str, str]],
    ) -> CallAttempt:
        """Wrap the answer's first LLM call as a `CallAttempt`.

        Any parse retries append their own behind it, so the
        list is every call the answer took, in order.

        :param response: Raw chat response (role-stripped).
        :param messages: The messages that call sent.
        :return: The first attempt.
        """
        return CallAttempt(
            metrics=LLMMetrics(**response["metrics"]),
            messages=messages,
        )

    def _finalize_response(
        self,
        response: dict,
        parsed: "BaseModel | None",
        idx: int = 0,
        attempts: "list[CallAttempt] | None" = None,
    ) -> AgentResponse:
        """Validate, wrap, and log a parsed LLM response.

        :param response: Raw chat response (role-stripped).
        :param parsed: Parsed model from the response, or
            None if parsing failed.
        :param idx: Optional index for batch tracking.
        :param attempts: Every LLM call the answer took, in
            order — the first, then one per parse retry. Their
            metrics add up into the response, because a parse
            retry is a whole extra call and reporting only the
            first understated exactly the answers that cost
            most; and each becomes its own call-log row, so the
            log counts the calls that were actually made.
        :return: Fully built AgentResponse.
        """
        response["model"] = self._parser.run_validators(parsed)
        response["idx"] = idx
        attempts = attempts or []
        if len(attempts) > 1:
            response["metrics"] = LLMMetrics.sum(
                [attempt.metrics for attempt in attempts]
            )
        agent_response = AgentResponse(**response)
        self._log_attempts(
            attempts, answered=agent_response.model is not None
        )
        return agent_response

    def _log_attempts(
        self,
        attempts: list[CallAttempt],
        *,
        answered: bool,
    ) -> None:
        """Emit one `CallRecord` per LLM call the answer took.

        A parse retry is a whole extra call to the provider, so
        it gets its own row. Folding them all into one made
        `CallStats.calls` report a single call where three were
        made, and divided three calls' seconds by one to get a
        per-model average duration.

        Every call but the last failed to parse, since the retry
        loop stops at the first that does — so only the last one
        can carry the answer's outcome, and the earlier ones are
        failures whatever became of it.

        :param attempts: The calls, in the order they were made.
        :param answered: Whether the answer ended with a usable
            model. False when the last retry still would not
            parse, and also when a validator rejected what did.
        """
        if self.call_logger is None:
            return
        trace_id = generate_trace_id()
        last = len(attempts) - 1
        for i, attempt in enumerate(attempts):
            succeeded = answered and i == last
            self._log_call(
                attempt.metrics,
                trace_id=trace_id,
                success=succeeded,
                messages=None if succeeded else attempt.messages,
            )

    def _emit_call_log(
        self,
        response: AgentResponse,
        error: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> None:
        """Emit one `CallRecord` for a single-call response.

        The batch and streaming paths, neither of which retries
        a failed parse, so their response is one LLM call and
        one row. `act` and `act_async` go through
        `_log_attempts` instead, since they can be more.

        :param response: The agent response.
        :param error: Optional error message.
        :param messages: The messages sent to the engine, kept
            on the record when parsing failed.
        """
        failed = response.model is None
        self._log_call(
            response.metrics,
            trace_id=generate_trace_id(),
            success=not failed,
            error=error,
            messages=messages if failed else None,
        )

    def _log_call(
        self,
        metrics: LLMMetrics,
        *,
        trace_id: str,
        success: bool,
        error: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> None:
        """Write one call's record to the call logger, if set.

        The prompt is recorded only when the call failed — see
        `CallRecord.prompt` for why a successful call carries
        none. A caller that passes no `messages` records none
        either way.

        :param metrics: That call's own metrics, never a total
            over several calls.
        :param trace_id: Shared by every call of one answer, so
            a consumer can group them.
        :param success: Whether this call produced the answer.
        :param error: Optional error message.
        :param messages: The messages this call sent, kept on
            the record when it failed.
        """
        if self.call_logger is None:
            return
        self.call_logger(
            CallRecord(
                trace_id=trace_id,
                run_id=self.run_id,
                recorded_at=now(),
                agent_name=self.name,
                agent_version=self.version,
                model_id=self.engine.model_id,
                success=success,
                tokens=metrics.tokens,
                prompt_tokens=metrics.prompt_tokens,
                cached_prompt_tokens=metrics.cached_prompt_tokens,
                total_duration=metrics.total_duration,
                cost_cents=metrics.cost_cents,
                cached=metrics.cached,
                model=metrics.model,
                retries=metrics.retries,
                retry_duration=metrics.retry_duration,
                error=error,
                prompt=(
                    json.dumps(messages, ensure_ascii=False)
                    if messages is not None
                    else None
                ),
            )
        )

    @staticmethod
    def _strip_role(response: dict) -> dict:
        """Remove the ``role`` key from a chat response dict.

        :param response: Raw chat response from the engine.
        :return: The same dict with ``role`` removed.
        """
        response.pop("role", None)
        return response

    def _validate_input(self, input_data) -> None:
        """Raise AgentConfigurationError for non-dict/list input.

        :param input_data: Raw input data to validate.
        :raises AgentConfigurationError: If input_data is not
            a dict or list of dicts.
        """
        if isinstance(input_data, dict):
            return
        if isinstance(input_data, list) and all(
            isinstance(item, dict) for item in input_data
        ):
            return
        raise AgentConfigurationError(
            "`input_data` must be a dict or list of dicts, "
            f"got {type(input_data).__name__}"
        )

    def _prepare_input(
        self,
        input_data: dict | list[dict],
        chat_kwargs: dict | None,
    ) -> tuple[list[dict[str, str]], dict]:
        """Validate, preprocess input, and build messages.

        :param input_data: Raw input data.
        :param chat_kwargs: Raw chat keyword arguments.
        :return: Tuple of (messages, sanitized chat_kwargs).
        """
        self._validate_input(input_data)
        chat_kwargs = self._sanitize_chat_kwargs(chat_kwargs)
        input_data = self._preprocess(input_data)
        messages = self._build_act_messages(input_data)
        return messages, chat_kwargs

    def _preprocess(self, input_data: Any) -> Any:
        """Apply the preprocessor if set.

        :param input_data: Raw input data.
        :return: Transformed input data, or original if no
            preprocessor is set.
        """
        if self.preprocessor is None:
            return input_data
        return self.preprocessor(input_data)

    def _sanitize_chat_kwargs(self, chat_kwargs: dict | None) -> dict:
        """
        Prepare `chat_kwargs` for an LLM call.
        Sets `think` to the agent's default and removes keys
        managed internally (`messages`, `temperature`,
        `output_format`). Users can override `think` via
        `chat_kwargs`.
        :param chat_kwargs: Raw keyword arguments, or None.
        :return: Sanitized copy of the keyword arguments.
        """
        chat_kwargs = dict(chat_kwargs) if chat_kwargs else {}
        chat_kwargs.setdefault("think", self.think)
        for k in ["messages", "temperature", "output_format"]:
            chat_kwargs.pop(k, None)
        return chat_kwargs
