"""Every measurement the framework declares must be produced.

This suite exists because of a failure that took months to
notice and that nothing else here could catch: a field is
declared on `LLMMetrics`, some code path never fills it, and the
default it falls back to is a *plausible value*. `cost_cents`
stayed `None` on every response ever produced; `total_duration`
came back as a hard `0.0` from one provider; a streamed call
skipped cost entirely; a guarded reply's second LLM call reported
no tokens at all. Nothing failed. Type checks passed, the suite
was green, and the numbers on screen were simply wrong.

The tests below are deliberately *structural* — they read the
declaration and check the producers against it, rather than
asserting one hand-written expectation. So adding a field to
`LLMMetrics`, a provider, or a response path fails here until
that new thing is wired up everywhere it belongs. That is the
point: the guard has to come from the declaration, because a
human remembering to update every producer is exactly what
failed.

See `.claude/rules/metrics-completeness.md`.
"""

import ast
import json
from dataclasses import fields as dataclass_fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import ollama
import pytest

import agents7
from agents7.agents.context_guard import (
    ContextGuard,
    GuardReason,
    GuardVerdict,
)
from agents7.call_log import CallRecord, CallStats, summarize_call_log
from agents7.llm.anthropic import AnthropicProvider
from agents7.llm.cache import CacheConfig, ResponseCache
from agents7.llm.dispatcher import EngineDispatcher
from agents7.llm.ollama import (
    parse_ollama_chat_response,
    parse_ollama_response,
)
from agents7.llm.openai_compat import OpenAICompatProvider
from agents7.llm.pricing import Pricing
from agents7.llm.retry import AttemptStats, RetryConfig, retry_sync
from agents7.llm.streaming import StreamResult
from agents7.models import AgentResponse, LLMMetrics

from helpers import (
    DUMMY_METRICS,
    chat_chunk,
    chat_response_dict,
    make_chatter,
    make_engine,
    make_guard,
    make_labeler,
)


# Fields the dispatcher adds after the provider returns, so a
# provider is not expected to produce them. Keep this list short
# and justified — every entry is a measurement that some *other*
# layer must then be shown to fill.
#
# `cost_cents` needs the engine's declared rates, which a provider
# knows nothing about. `cached` is a fact about the dispatch
# pipeline rather than about the response: only the layer that
# looked the call up in the cache can say whether it was a hit.
# `retries` and `retry_duration` are facts about the retry loop,
# which sits above the provider — a provider call either returns
# or raises, and knows nothing of the attempts before it.
# All four are covered by the dispatcher tests below.
DISPATCHER_OWNED = {
    "cost_cents",
    "cached",
    "retries",
    "retry_duration",
}

PROVIDER_OWNED = set(LLMMetrics.model_fields) - DISPATCHER_OWNED


# ── The declaration itself ─────────────────────────────────────────────────


def test_every_metric_has_a_producer_or_an_exemption():
    """No field may quietly belong to nobody.

    A field that is neither produced by a provider nor added by
    the dispatcher is one nothing fills — which is the whole bug
    class. Adding a metric fails here until it is classified.
    """
    classified = PROVIDER_OWNED | DISPATCHER_OWNED

    assert set(LLMMetrics.model_fields) == classified


# ── Providers ──────────────────────────────────────────────────────────────


def _ollama_chat_response(model: str = "qwen3"):
    msg = SimpleNamespace(
        content="hola", role="assistant", thinking=None, tool_calls=None
    )
    return SimpleNamespace(
        message=msg,
        eval_count=9,
        prompt_eval_count=6239,
        total_duration=1_500_000_000,
        model=model,
    )


def _ollama_generate_response():
    return SimpleNamespace(
        response="hola",
        thinking=None,
        eval_count=9,
        prompt_eval_count=6239,
        total_duration=1_500_000_000,
        model="qwen3",
    )


def _openai_response():
    usage = SimpleNamespace(
        prompt_tokens=6239,
        completion_tokens=9,
        total_tokens=6248,
        prompt_cache_hit_tokens=6144,
    )
    msg = SimpleNamespace(content="hola", role="assistant")
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage=usage,
        model="deepseek-chat",
    )


def _anthropic_response():
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hola")],
        role="assistant",
        usage=SimpleNamespace(input_tokens=6239, output_tokens=9),
        model="claude-sonnet-4-20250514",
    )


@pytest.mark.parametrize(
    "name, produce",
    [
        (
            "ollama.chat",
            lambda: parse_ollama_chat_response(_ollama_chat_response()),
        ),
        (
            "ollama.generate",
            lambda: parse_ollama_response(_ollama_generate_response()),
        ),
        (
            "openai_compat",
            lambda: OpenAICompatProvider._parse_response(
                _openai_response(), elapsed=1.5
            ),
        ),
        (
            "anthropic",
            lambda: AnthropicProvider._parse_response(
                _anthropic_response(), start_time=0.0
            ),
        ),
    ],
)
def test_provider_reports_every_metric_it_owns(name, produce):
    """Each provider fills every field it is responsible for.

    A provider that omits one leaves the model's default in
    place, and every default here is a legitimate-looking value —
    which is why the omission was invisible.
    """
    metrics = produce()["metrics"]
    missing = PROVIDER_OWNED - set(metrics)

    assert not missing, f"{name} does not report: {sorted(missing)}"


@pytest.mark.parametrize(
    "name, produce",
    [
        (
            "ollama.chat",
            lambda: parse_ollama_chat_response(_ollama_chat_response()),
        ),
        (
            "openai_compat",
            lambda: OpenAICompatProvider._parse_response(
                _openai_response(), elapsed=1.5
            ),
        ),
        (
            "anthropic",
            lambda: AnthropicProvider._parse_response(
                _anthropic_response(), start_time=0.0
            ),
        ),
    ],
)
def test_provider_reports_a_real_duration(name, produce):
    """A duration of exactly zero is a missing measurement.

    `openai_compat` returned a hard `0.0` for months. It is a
    valid float, so nothing complained, and it read as a call
    that took no time.
    """
    duration = produce()["metrics"]["total_duration"]

    assert duration > 0, f"{name} reports a zero duration"


def test_a_cached_count_never_exceeds_the_prompt_total():
    """Cached tokens are a subset, on every provider.

    Anthropic reports cache reads *outside* `input_tokens`, the
    OpenAI-compatible providers report them inside it. The cost
    estimate subtracts one from the other, so a provider that
    breaks this invariant silently produces a negative fresh-token
    count and undercharges.
    """
    for produce in (
        lambda: OpenAICompatProvider._parse_response(
            _openai_response(), elapsed=1.5
        ),
        lambda: AnthropicProvider._parse_response(
            _anthropic_response(), start_time=0.0
        ),
    ):
        m = produce()["metrics"]
        assert m["cached_prompt_tokens"] <= m["prompt_tokens"]


# ── Response paths ─────────────────────────────────────────────────────────


def test_streaming_reports_the_same_metrics_as_a_plain_call():
    """A streamed turn is not a lesser turn.

    Streaming built its own metrics dict and drifted: it reported
    no cached count, a zero duration, and never had cost injected
    at all — on the very path a chat UI uses.
    """
    plain = set(
        parse_ollama_chat_response(_ollama_chat_response())["metrics"]
    )

    stream = StreamResult(
        iter(
            [
                chat_chunk(content="ho"),
                chat_chunk(
                    done=True,
                    eval_count=9,
                    prompt_eval_count=6239,
                    cached_prompt_tokens=6144,
                    total_duration=1_500_000_000,
                ),
            ]
        ),
        "chat",
    )
    list(stream)

    assert set(stream.metrics) == plain


# ── Downstream consumers ───────────────────────────────────────────────────


def test_call_record_carries_every_metric():
    """The trace surface mirrors the measurement surface.

    `CallRecord` copied three of the five fields, so an eval run's
    call log could not report what the run cost however the engine
    was priced.
    """
    recorded = {f.name for f in dataclass_fields(CallRecord)}
    missing = set(LLMMetrics.model_fields) - recorded

    assert not missing, f"CallRecord does not record: {sorted(missing)}"


# A metric whose aggregate goes under a different name in
# `CallStats`, because summing it means something else. Every
# other field keeps its name. An entry here is a claim that the
# aggregate exists under that name, which the test then checks.
STATS_RENAMES = {
    # A boolean cannot be summed, so the aggregate counts the
    # calls that were hits instead.
    "cached": "cached_calls",
    # A name cannot be summed either, so the aggregate is the
    # distinct names. More than one entry means the bucket's
    # calls were not all answered by the same model.
    "model": "models",
}


def test_call_stats_aggregates_every_metric():
    """A summary that drops a field cannot report it either.

    `CallStats` carried three of the five fields for as long as
    `CallRecord` did, so summarising a log lost exactly what the
    log had just started recording.
    """
    aggregated = {f.name for f in dataclass_fields(CallStats)}
    expected = {
        STATS_RENAMES.get(name, name) for name in LLMMetrics.model_fields
    }
    missing = expected - aggregated

    assert not missing, f"CallStats does not aggregate: {sorted(missing)}"


def test_summing_metrics_carries_every_field():
    """The one aggregator must not drop a field either.

    A guarded turn, a tool loop, a pipeline and the UI's guard
    summary all add calls together. Each summed field by field
    until `LLMMetrics.sum` existed, and a new field had to be
    remembered in four places — so this checks the one place.
    """
    part = LLMMetrics(
        tokens=9,
        prompt_tokens=6239,
        cached_prompt_tokens=6144,
        total_duration=1.5,
        cost_cents=0.0168,
        cached=True,
        model="ollama::qwen3",
        retries=1,
        retry_duration=2.0,
    )

    total = LLMMetrics.sum([part, part])

    # Every field carries something from the parts. A field the
    # sum forgot falls back to its default, and every default
    # here is a plausible-looking value.
    default = LLMMetrics(tokens=0, prompt_tokens=0, total_duration=0.0)
    dropped = [
        name
        for name in LLMMetrics.model_fields
        if getattr(total, name) == getattr(default, name)
    ]

    assert not dropped, f"LLMMetrics.sum drops: {sorted(dropped)}"


def test_a_response_made_entirely_of_cache_hits_costs_zero():
    """Zero and unpriced are different answers, and both occur.

    A priced engine whose every call hit the cache really did
    spend nothing, so the total is `0.0`. An engine with no rates
    at all stays `None`. Summing on truthiness conflated the two
    and reported the free run as unpriced.
    """
    hit = LLMMetrics(
        tokens=9,
        prompt_tokens=6239,
        total_duration=0.0004,
        cost_cents=0.0,
        cached=True,
    )
    unpriced = LLMMetrics(tokens=9, prompt_tokens=6239, total_duration=1.5)

    assert LLMMetrics.sum([hit, hit]).cost_cents == 0.0
    assert LLMMetrics.sum([unpriced, unpriced]).cost_cents is None
    assert LLMMetrics.sum([]).cost_cents is None


def test_a_streamed_call_reaches_the_call_log():
    """A streamed turn is logged like a plain one.

    `act_stream` returned an `AgentStreamResult` and never
    emitted a record, so a run made entirely of streamed calls
    wrote an empty log — no tokens, no duration, no cost, no
    failures. Nothing failed, because an empty file looks like a
    run that made no calls.
    """
    records = []
    agent = make_labeler(engine=make_engine(), call_logger=records.append)
    agent.engine.llm_provider.chat_stream.return_value = StreamResult(
        iter(
            [
                chat_chunk(content='{"value": "hi"}'),
                chat_chunk(
                    done=True,
                    eval_count=9,
                    prompt_eval_count=6239,
                    total_duration=1_500_000_000,
                ),
            ]
        ),
        "chat",
    )

    stream = agent.act_stream({"text": "hi"})
    list(stream)

    assert len(records) == 1
    assert records[0].success is True
    assert records[0].tokens == 9
    assert records[0].prompt_tokens == 6239
    assert records[0].total_duration == 1.5


# ── The dispatcher's own fields ────────────────────────────────────────────


def _priced_dispatcher(cache_config=None):
    return EngineDispatcher(
        primary=None,
        fallback=None,
        retry_config=None,
        cache=ResponseCache(cache_config or CacheConfig()),
        pricing=Pricing(input=100.0, output=100.0),
        model_id="ollama::test-model",
    )


def test_dispatcher_states_every_field_it_owns_on_a_live_call():
    """A live call says what it is, rather than defaulting to it.

    `cached=False` has to be written into the metrics dict, not
    left to the model default — the whole bug class is a field
    nothing fills falling back to a plausible-looking value.
    """
    metrics = {"tokens": 9, "prompt_tokens": 6239}

    _priced_dispatcher()._finalize_metrics(metrics)

    missing = DISPATCHER_OWNED - set(metrics)
    assert not missing, f"dispatcher does not state: {sorted(missing)}"


def test_a_cache_hit_reports_its_own_numbers_not_the_stored_ones():
    """A hit costs nothing and takes microseconds.

    Replaying the stored duration and cost made a cached call
    report time it did not take and money it did not spend, which
    silently inflated every total that included one.
    """
    stored = {
        "response": "hola",
        "metrics": {
            "tokens": 9,
            "prompt_tokens": 6239,
            "total_duration": 1.5,
            "cost_cents": 62.48,
            "cached": False,
        },
    }

    hit = _priced_dispatcher()._as_cache_hit(stored, elapsed=0.0004)[
        "metrics"
    ]

    assert hit["cached"] is True
    assert hit["cost_cents"] == 0.0
    assert hit["total_duration"] == 0.0004
    # The tokens describe the response being returned, so they
    # stay; `cached` is what says they were not billed again.
    assert hit["tokens"] == 9
    assert hit["prompt_tokens"] == 6239
    # The stored entry is handed back by reference on every later
    # hit, so re-measuring it must never write through to it.
    assert stored["metrics"]["total_duration"] == 1.5
    assert stored["metrics"]["cost_cents"] == 62.48


def test_a_retried_call_says_so():
    """A run that fought the provider must not look clean.

    A call retried three times reached the caller identical to one
    that succeeded first time, so a run that spent minutes on
    backoff was indistinguishable from a fast one — with nothing
    on the response to explain where the wall clock went.
    """
    stats = AttemptStats()
    attempts = []

    def flaky(**kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise ollama.RequestError("connection refused")
        return {"metrics": {"tokens": 9, "prompt_tokens": 6239}}

    result = retry_sync(
        flaky,
        RetryConfig(max_retries=3, base_delay=0.001),
        stats,
    )
    _priced_dispatcher()._finalize_result(result, stats)
    metrics = result["metrics"]

    assert metrics["retries"] == 2
    assert metrics["retry_duration"] > 0


def test_a_clean_call_reports_no_retries():
    """Zero here is a measurement, not an absence.

    It has to be written into the dict by the layer that ran the
    retry loop — left to the model default it would read the same
    on a path that never looked.
    """
    metrics = {"tokens": 9, "prompt_tokens": 6239}

    _priced_dispatcher()._finalize_metrics(metrics, AttemptStats())

    assert metrics["retries"] == 0
    assert metrics["retry_duration"] == 0.0


def test_a_cache_hit_does_not_replay_the_stored_retries():
    """A lookup cannot fail, so it cannot have retried.

    The stored entry is the finished response of a call that may
    well have been retried, and returning its count would charge
    a later dictionary lookup with the original call's struggle.
    """
    stored = {
        "response": "hola",
        "metrics": {
            "tokens": 9,
            "prompt_tokens": 6239,
            "retries": 3,
            "retry_duration": 7.0,
        },
    }

    hit = _priced_dispatcher()._as_cache_hit(stored, elapsed=0.0004)[
        "metrics"
    ]

    assert hit["retries"] == 0
    assert hit["retry_duration"] == 0.0
    assert stored["metrics"]["retries"] == 3


def test_an_unpriced_cache_hit_stays_unpriced():
    """Zero is a price, and an unpriced engine has none.

    Reporting `0.0` here would turn a whole unpriced run's `None`
    total into `0.0` the moment one call hit the cache, which
    reads as "this run was free" rather than "nobody priced it".
    """
    dispatcher = EngineDispatcher(
        primary=None,
        fallback=None,
        retry_config=None,
        cache=ResponseCache(CacheConfig()),
        pricing=None,
        model_id="ollama::test-model",
    )
    stored = {"response": "hola", "metrics": {"tokens": 9}}

    hit = dispatcher._as_cache_hit(stored, elapsed=0.0004)["metrics"]

    assert hit["cost_cents"] is None
    assert hit["cached"] is True


# ── A chat turn's calls ────────────────────────────────────────────────────


def _verdict(value: bool, reason: str) -> AgentResponse:
    """A parsed guard response, accepted or rejected."""
    verdict = GuardVerdict(
        value=value,
        reason=GuardReason[reason],
        fix="Soy David." if not value else None,
    )
    return AgentResponse(
        response="{}",
        model=verdict,
        thinking=None,
        metrics=DUMMY_METRICS,
    )


def _accept() -> AgentResponse:
    """The guard passing a response."""
    return _verdict(True, "IN_CHARACTER")


def _reject() -> AgentResponse:
    """The guard rejecting a response."""
    return _verdict(False, "BROKE_CHARACTER")


def _guarded_agent(records: list, guard_engine=None):
    """A guarded chatter that logs into *records*."""
    from agents7.agents.guarded import GuardedChatter

    return GuardedChatter(
        name="guarded-david",
        version=1,
        engine=make_engine(),
        system_prompt="You are David.",
        guard=make_guard(
            engine=guard_engine or make_engine(),
            persona_summary="David, Spanish journalist",
        ),
        call_logger=records.append,
    )


def test_a_chat_turn_reaches_the_call_log():
    """A chat agent produced no records at all.

    `Chatter` carried no `call_logger`, so every conversational
    agent was absent from the call log — a run made entirely of
    chat turns wrote an empty file, and an eval over one could
    report neither its tokens nor its cost.
    """
    records = []
    agent = make_chatter(engine=make_engine(), call_logger=records.append)
    agent.engine.llm_provider.chat.return_value = chat_response_dict("hola")

    agent.chat("hey")

    assert len(records) == 1
    assert records[0].agent_name == "test-chatter"
    assert records[0].tokens == DUMMY_METRICS["tokens"]
    assert records[0].success is True


def test_a_guarded_turn_logs_the_guard_call_as_its_own_record():
    """The guard is the second half of what a turn costs.

    Its metrics reached the `GuardEvent` and stopped there, so a
    run's reported cost excluded every guard call — on one agent,
    46% of the real bill. It gets its own record rather than
    being folded into the reply's, because a guard usually runs
    a different model and one record cannot report two.
    """
    records = []
    guard_engine = make_engine(model_name="guard-model")
    agent = _guarded_agent(records, guard_engine=guard_engine)
    agent.engine.llm_provider.chat.return_value = chat_response_dict("hola")
    with patch.object(ContextGuard, "act", return_value=_accept()):
        agent.chat("hey")

    assert len(records) == 2
    reply, guard = records
    assert reply.agent_name == "guarded-david"
    assert reply.model_id == agent.engine.model_id
    assert guard.agent_name == "test-guard"
    assert guard.model_id == guard_engine.model_id
    assert guard.tokens == DUMMY_METRICS["tokens"]


def test_every_call_in_a_turn_shares_the_turns_trace_id():
    """One turn, one id, however many calls it took."""
    records = []
    agent = _guarded_agent(records)
    agent.engine.llm_provider.chat.return_value = chat_response_dict("hola")
    with patch.object(ContextGuard, "act", return_value=_accept()):
        agent.chat("hey")
        agent.chat("again")

    first, second = records[:2], records[2:]
    assert len({r.trace_id for r in first}) == 1
    assert len({r.trace_id for r in second}) == 1
    assert first[0].trace_id != second[0].trace_id


def test_a_rejected_turn_logs_every_call_it_made():
    """A retried turn cost more and must say so.

    The guard loop discarded the regenerated calls, so a turn
    rejected once looked in the log exactly like one accepted
    first time — while having cost twice as much.
    """
    records = []
    agent = _guarded_agent(records)
    agent.engine.llm_provider.chat.side_effect = [
        chat_response_dict("I am an AI."),
        chat_response_dict("Soy David."),
    ]
    with patch.object(
        ContextGuard,
        "act",
        side_effect=[_reject(), _accept()],
    ):
        agent.chat("who are you?")

    # reply, guard, regenerated reply, guard again
    assert len(records) == 4
    assert [r.agent_name for r in records] == [
        "guarded-david",
        "test-guard",
        "guarded-david",
        "test-guard",
    ]


def test_an_unparseable_guard_call_records_its_prompt():
    """A failed call is the one whose prompt cannot be re-derived.

    The guard's input is built from the conversation at that
    moment, so a re-run at temperature does not reproduce it.
    """
    records = []
    agent = _guarded_agent(records)
    agent.engine.llm_provider.chat.return_value = chat_response_dict("hola")
    unparseable = AgentResponse(
        response="not json",
        model=None,
        thinking=None,
        metrics=DUMMY_METRICS,
    )
    with patch.object(ContextGuard, "act", return_value=unparseable):
        agent.chat("hey")

    guard = records[1]
    assert guard.success is False
    assert json.loads(guard.prompt)["assistant_response"] == "hola"
    # The reply itself succeeded, so it carries none.
    assert records[0].success is True
    assert records[0].prompt is None


def test_a_parse_retry_is_its_own_record():
    """The same invariant, on the structured-output path.

    A parse retry is a whole extra LLM call — it succeeded, it
    just answered off-schema. Its tokens and its cost reached
    the response, but all three calls were written as a single
    record, so `CallStats.calls` said one where three were made
    and a per-model average duration divided three calls'
    seconds by one. It is the labeler's version of what the
    guard loop got wrong above, and it is fixed the same way:
    one record per call, sharing the answer's trace id.
    """
    records = []
    engine = make_engine()
    agent = make_labeler(
        engine=engine,
        max_parse_retries=2,
        call_logger=records.append,
    )
    engine.llm_provider.chat.side_effect = [
        chat_response_dict("not json"),
        chat_response_dict("not json"),
        chat_response_dict(json.dumps({"value": "finally"})),
    ]

    agent.act({"text": "hi"})

    assert len(records) == 3
    assert [r.success for r in records] == [False, False, True]
    assert len({r.trace_id for r in records}) == 1
    # Each row reports its own call, never the total, so summing
    # the log does not count the same tokens three times.
    assert [r.tokens for r in records] == [DUMMY_METRICS["tokens"]] * 3


@pytest.mark.parametrize("required", ["run_id", "recorded_at"])
def test_every_call_record_producer_names(required):
    """A field one producer fills is the failure this rule names.

    Both are knowable wherever a `CallRecord` is built and
    neither comes from the provider, so neither is caught by the
    per-provider checks above. `run_id` groups the turns of a
    conversation or an eval run — a chatter has its
    `conversation_id`, the other agents take one from the caller.
    `recorded_at` says when the call happened, which is what lets
    a log be read against a date rather than only in order.

    Read at the source rather than exercised, so a producer added
    later fails here without needing its own test.
    """
    root = Path(agents7.__file__).parent
    missing = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", None) != "CallRecord":
                continue
            named = {kw.arg for kw in node.keywords}
            # A `**`-unpacked construction is a row being read
            # back from a log, not a call being measured — it
            # must carry through whatever was stored, including
            # nothing.
            if None in named:
                continue
            if required not in named:
                missing.append(f"{path.name}:{node.lineno}")
    assert not missing, f"CallRecord built without {required}: {missing}"


def test_a_record_read_back_from_an_old_log_is_not_stamped_today():
    """An absent timestamp must not be rendered as a real one.

    `recorded_at` is stamped by each producer rather than by a
    field default, so a row from a log written before the field
    existed reads as unstamped. A default factory would have
    given it the moment it was read, which is the on-screen form
    of the bug this whole rule is about.
    """
    old_line = json.dumps(
        {
            "trace_id": "abc",
            "agent_name": "Extractor",
            "agent_version": 1,
            "model_id": "ollama::qwen3",
            "success": True,
        }
    )

    record = CallRecord(**json.loads(old_line))

    assert record.recorded_at is None
    assert record.run_id is None


def test_a_chat_turn_records_the_conversation_it_belongs_to():
    """`trace_id` groups a turn; nothing grouped the turns.

    A chatter always knows which conversation it is in, so the
    id was available at the moment the record was written and
    was dropped. Without it a log can say what a turn cost and
    not what a conversation cost.
    """
    records = []
    agent = make_chatter(
        engine=make_engine(),
        call_logger=records.append,
        conversation_id="cold-read",
    )
    agent.engine.llm_provider.chat.return_value = chat_response_dict("hola")

    agent.chat("hey")
    agent.chat("again")

    assert [r.run_id for r in records] == ["cold-read", "cold-read"]
    # The turns are still distinguishable within the run.
    assert len({r.trace_id for r in records}) == 2


def test_a_guarded_turns_guard_call_shares_the_run():
    """The guard's record is its own row, in the same run.

    It carries a different agent name and usually a different
    model, so it cannot be folded into the reply's record — but
    it is part of the same conversation and groups with it.
    """
    records = []
    agent = _guarded_agent(records)
    agent._set_attr("conversation_id", "warm-read")
    agent.engine.llm_provider.chat.return_value = chat_response_dict("hola")
    with patch.object(ContextGuard, "act", return_value=_accept()):
        agent.chat("hey")

    assert {r.run_id for r in records} == {"warm-read"}


def test_a_labeler_carries_the_run_the_caller_named():
    """Every attempt of one answer belongs to the same run."""
    records = []
    engine = make_engine()
    agent = make_labeler(
        engine=engine,
        max_parse_retries=1,
        call_logger=records.append,
        run_id="sweep-3",
    )
    engine.llm_provider.chat.side_effect = [
        chat_response_dict("not json"),
        chat_response_dict(json.dumps({"value": "finally"})),
    ]

    agent.act({"text": "hi"})

    assert [r.run_id for r in records] == ["sweep-3", "sweep-3"]


def test_an_unnamed_run_stays_unnamed():
    """None means nobody named a run, not one run called None.

    The same distinction `cost_cents` draws between unpriced and
    free: a default that looks like a value is what makes a
    missing measurement invisible.
    """
    records = []
    agent = make_labeler(engine=make_engine(), call_logger=records.append)
    agent.engine.llm_provider.chat.return_value = chat_response_dict(
        json.dumps({"value": "ok"})
    )

    agent.act({"text": "hi"})

    assert records[0].run_id is None


# ── Which model actually answered ──────────────────────────────────────────
#
# `model` is the one field whose whole purpose is to differ from
# what the config says. Two ways it does, and both were invisible
# until v5.4.0: a provider resolving an alias to something else,
# and a `fallback_model` answering after the primary failed. The
# guards above prove the field is declared and carried; these
# prove it is filled with what answered rather than with what was
# asked for.


def test_every_provider_names_the_model_that_answered():
    """Not one of them may leave it to the default.

    A `None` here is indistinguishable from a provider that
    named nothing back, which is the shape of every bug this
    suite exists to catch.
    """
    produced = {
        "ollama.chat": parse_ollama_chat_response(_ollama_chat_response()),
        "ollama.generate": parse_ollama_response(_ollama_generate_response()),
        "openai_compat": OpenAICompatProvider._parse_response(
            _openai_response(), elapsed=1.5
        ),
        "anthropic": AnthropicProvider._parse_response(
            _anthropic_response(), start_time=0.0
        ),
    }
    unnamed = [
        name
        for name, result in produced.items()
        if not result["metrics"].get("model")
    ]

    assert not unnamed, f"these name no model: {sorted(unnamed)}"


def test_a_provider_reports_what_it_ran_not_what_was_asked_for():
    """An alias must not be reported as the model that ran.

    Ask DeepSeek for `deepseek-chat` and the reply says
    `deepseek-v4-flash`. Reading the configured name instead
    would make the two indistinguishable, and a run could not
    say what actually served it.
    """
    response = _openai_response()
    response.model = "deepseek-v4-flash"

    metrics = OpenAICompatProvider._parse_response(response, elapsed=1.5)[
        "metrics"
    ]

    assert metrics["model"] == "openai_compat::deepseek-v4-flash"


def test_a_fallback_answer_names_the_fallback():
    """A rescued response says which model rescued it.

    The engine's own `model_id` stays the primary's — it is what
    the agent was configured with — so a consumer reading only
    that logged a run half-answered by the fallback entirely
    against the model that failed.
    """
    primary = Mock()
    primary.chat.side_effect = ollama.RequestError("connection refused")
    fallback = Mock()
    fallback.chat.return_value = parse_ollama_chat_response(
        _ollama_chat_response(model="qwen2.5:3b")
    )
    dispatcher = EngineDispatcher(
        primary=primary,
        fallback=fallback,
        retry_config=RetryConfig(max_retries=1, base_delay=0.001),
        cache=None,
        pricing=None,
        model_id="ollama::gemma3:12b",
    )

    with patch("agents7.llm.retry.time.sleep"):
        result = dispatcher.execute(
            "chat",
            primary.chat,
            "messages",
            [],
            temperature=0,
            seed=None,
            max_tokens=-1,
            output_format=None,
            think=False,
        )

    assert result["metrics"]["model"] == "ollama::qwen2.5:3b"


def test_a_call_record_names_the_model_that_answered():
    """The log carries it, not just the configured name.

    Declaring the field and leaving every producer to the default
    is the exact failure `cost_cents` had for months, so this
    checks a real agent call end to end rather than the shape of
    the dataclass.
    """
    records = []
    agent = make_labeler(engine=make_engine(), call_logger=records.append)
    agent.engine.llm_provider.chat.return_value = {
        "response": '{"value": "hi"}',
        "metrics": {**DUMMY_METRICS, "model": "ollama::qwen2.5:3b"},
    }

    agent.act({"text": "hi"})

    assert len(records) == 1
    # What it was configured with, and what answered — both, and
    # different, which is the whole point of carrying two.
    assert records[0].model_id == "ollama::test-model"
    assert records[0].model == "ollama::qwen2.5:3b"


def test_a_summary_names_every_model_that_answered():
    """One bucket, two models, and the summary says so.

    A name has no sum, so `CallStats` collects the distinct
    names. Two entries is what a run partly served by a fallback
    looks like, and reporting one would hide it.
    """
    common = dict(
        trace_id="t",
        agent_name="a",
        agent_version=1,
        model_id="ollama::gemma3:12b",
        success=True,
    )
    summary = summarize_call_log(
        [
            CallRecord(**common, model="ollama::gemma3:12b"),
            CallRecord(**common, model="ollama::qwen2.5:3b"),
        ]
    )

    assert summary.overall.models == (
        "ollama::gemma3:12b",
        "ollama::qwen2.5:3b",
    )
