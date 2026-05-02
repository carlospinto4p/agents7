"""Comparison runner — run two prompt versions side-by-side."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Callable

from agents7.models.base import AgentResponse
from agents7.ui.comparison.models import PromptVersion
from agents7.ui.helpers import build_agent_from_state
from agents7.ui.state import AgentBuilderState


@dataclass
class ComparisonResult:
    """Result of running a single input against two versions.

    Fields:
        input_data: The input dict sent to both agents.
        output_a: Parsed model output from version A, or `None`.
        output_b: Parsed model output from version B, or `None`.
        metrics_a: LLM metrics dict from version A.
        metrics_b: LLM metrics dict from version B.
        error_a: Error message from version A, or `None`.
        error_b: Error message from version B, or `None`.
        outputs_match: Whether both outputs are identical.
    """

    input_data: dict
    output_a: dict | None = None
    output_b: dict | None = None
    metrics_a: dict = field(default_factory=dict)
    metrics_b: dict = field(default_factory=dict)
    error_a: str | None = None
    error_b: str | None = None
    outputs_match: bool = False


def _build_version_state(
    base_state: AgentBuilderState,
    version: PromptVersion,
) -> AgentBuilderState:
    """Create a copy of *base_state* with version overrides.

    Replaces `system_prompt`, `examples`, and `model_name`
    from the given `PromptVersion`, preserving all other
    fields (agent type, output model definition, etc.).
    """
    return replace(
        base_state,
        system_prompt=version.system_prompt,
        examples=version.examples,
        model_name=version.model_name,
    )


def _run_single(
    agent, input_data: dict
) -> tuple[dict | None, dict, str | None]:
    """Run a single input through an agent.

    Returns:
        Tuple of (output_dict, metrics_dict, error_string).
    """
    try:
        response: AgentResponse = agent.act(input_data)
        output = (
            response.model.model_dump()
            if response.model is not None
            else None
        )
        metrics = response.metrics.model_dump()
        return output, metrics, None
    except Exception as e:
        return None, {}, str(e)


def run_comparison(
    base_state: AgentBuilderState,
    version_a: PromptVersion,
    version_b: PromptVersion,
    inputs: list[dict],
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[ComparisonResult]:
    """Run *inputs* against two prompt versions and collect
    paired results.

    Builds one agent per version from *base_state* (using
    `build_agent_from_state`), then iterates over *inputs*,
    calling each agent and comparing outputs.

    Args:
        base_state: The current `AgentBuilderState` (defines
            agent type and output model).
        version_a: First prompt version.
        version_b: Second prompt version.
        inputs: List of input dicts.
        progress_callback: Optional `(current, total)`
            callable invoked after each input pair.

    Returns:
        List of `ComparisonResult`, one per input.
    """
    state_a = _build_version_state(base_state, version_a)
    state_b = _build_version_state(base_state, version_b)

    agent_a = build_agent_from_state(state_a)
    agent_b = build_agent_from_state(state_b)

    results: list[ComparisonResult] = []
    total = len(inputs)

    with ThreadPoolExecutor(max_workers=2) as pool:
        for idx, input_data in enumerate(inputs):
            fut_a = pool.submit(_run_single, agent_a, input_data)
            fut_b = pool.submit(_run_single, agent_b, input_data)
            out_a, met_a, err_a = fut_a.result()
            out_b, met_b, err_b = fut_b.result()

            match = out_a is not None and out_b is not None and out_a == out_b

            results.append(
                ComparisonResult(
                    input_data=input_data,
                    output_a=out_a,
                    output_b=out_b,
                    metrics_a=met_a,
                    metrics_b=met_b,
                    error_a=err_a,
                    error_b=err_b,
                    outputs_match=match,
                )
            )

            if progress_callback is not None:
                progress_callback(idx + 1, total)

    return results
