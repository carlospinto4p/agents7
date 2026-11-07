"""Unit tests for per-conversation context params."""

from unittest.mock import patch

import pytest
import yaml

from agents7.agents.chatter import Chatter
from agents7.agents.context import (
    ContextParam,
    spanish_time_of_day,
    spanish_weekday,
    time_of_day,
)
from agents7.eval.style import ConversationScript, StyleContract
from agents7.eval.style_history import fingerprint_run
from agents7.exceptions import (
    AgentConfigurationError,
    PromptParameterError,
)
from agents7.llm import LLMEngine

from helpers import chat_response_dict, make_chatter, make_labeler


def _resolver_value() -> str:
    """Module-level resolver, importable by dotted path."""
    return "resolved"


def _boom() -> str:
    """Resolver that always raises."""
    raise RuntimeError("no clock")


# --- ContextParam ---------------------------------------------


def test_context_param_defaults():
    """A bare declaration carries an empty default."""
    param = ContextParam(name="nick")
    assert param.default == ""
    assert param.description == ""
    assert param.resolver is None
    assert param.resolve() == ""


def test_context_param_resolver_wins_over_default():
    """A declared resolver is called instead of the default."""
    param = ContextParam(
        name="nick", default="ella", resolver=_resolver_value
    )
    assert param.resolve() == "resolved"


def test_context_param_resolver_failure_falls_back():
    """A raising resolver falls back to the default."""
    param = ContextParam(name="nick", default="ella", resolver=_boom)
    assert param.resolve() == "ella"


def test_context_param_rejects_empty_name():
    """An unnamed declaration is refused."""
    with pytest.raises(PromptParameterError):
        ContextParam(name="")


def test_context_param_rejects_non_string_default():
    """A non-string default is refused."""
    with pytest.raises(PromptParameterError):
        ContextParam(name="age", default=32)


def test_context_param_rejects_non_callable_resolver():
    """A resolver that cannot be called is refused."""
    with pytest.raises(PromptParameterError):
        ContextParam(name="nick", resolver="not_callable")


# --- Prompt resolution ----------------------------------------


def test_declared_default_substituted():
    """An unsupplied param resolves to its default."""
    agent = make_chatter(
        system_prompt="Hablas con $nick.",
        context_params=(ContextParam(name="nick", default="ella"),),
    )
    assert agent.resolved_system_prompt == "Hablas con ella."


def test_supplied_value_wins_over_default():
    """set_context replaces the declared default."""
    agent = make_chatter(
        system_prompt="Hablas con $nick.",
        context_params=(ContextParam(name="nick", default="ella"),),
    )
    agent.set_context(nick="Monica28")
    assert agent.resolved_system_prompt == "Hablas con Monica28."


def test_supplied_value_wins_over_resolver():
    """A caller value takes precedence over the resolver."""
    agent = make_chatter(
        system_prompt="Es $moment.",
        context_params=(
            ContextParam(name="moment", default="", resolver=_resolver_value),
        ),
    )
    assert agent.resolved_system_prompt == "Es resolved."
    agent.set_context(moment="de madrugada")
    assert agent.resolved_system_prompt == "Es de madrugada."


def test_resolver_recomputed_every_read():
    """The resolver runs again on each prompt build."""
    calls = []

    def counter() -> str:
        calls.append(1)
        return str(len(calls))

    agent = make_chatter(
        system_prompt="n=$n",
        context_params=(ContextParam(name="n", resolver=counter),),
    )
    assert agent.resolved_system_prompt == "n=1"
    assert agent.resolved_system_prompt == "n=2"


def test_declared_param_never_leaks_placeholder():
    """A declared name is never left as the literal $name."""
    agent = make_chatter(
        system_prompt="Hola $nick.",
        context_params=(ContextParam(name="nick"),),
    )
    assert "$nick" not in agent.resolved_system_prompt


def test_undeclared_placeholder_still_left_alone():
    """Undeclared placeholders keep the existing behaviour."""
    agent = make_chatter(system_prompt="Hola $stranger.")
    assert agent.resolved_system_prompt == "Hola $stranger."


def test_prompt_params_still_work_alongside():
    """Fixed prompt_params and context_params coexist."""
    agent = make_chatter(
        system_prompt="$city, con $nick.",
        prompt_params={"city": "Sevilla"},
        context_params=(ContextParam(name="nick", default="ella"),),
    )
    assert agent.resolved_system_prompt == "Sevilla, con ella."


@patch.object(LLMEngine, "chat")
def test_context_survives_a_chat_turn(mock_chat):
    """Supplied values persist across messages."""
    mock_chat.return_value = chat_response_dict("hola")
    agent = make_chatter(
        system_prompt="Hablas con $nick.",
        context_params=(ContextParam(name="nick", default="ella"),),
    )
    agent.set_context(nick="Monica28")

    agent.chat("hola")
    agent.chat("qué tal")

    prompts = [
        call.kwargs["messages"][0]["content"]
        for call in mock_chat.call_args_list
    ]
    assert prompts == ["Hablas con Monica28."] * 2


def test_context_params_available_on_a_labeler():
    """The mechanism lives on BaseAgent, not only Chatter."""
    agent = make_labeler(
        system_prompt="Clasifica para $nick.",
        context_params=(ContextParam(name="nick", default="ella"),),
    )
    assert agent.resolved_system_prompt == "Clasifica para ella."


# --- set_context ----------------------------------------------


def test_set_context_returns_the_agent():
    """set_context chains, returning the same instance."""
    agent = make_chatter(
        system_prompt="$nick",
        context_params=(ContextParam(name="nick"),),
    )
    assert agent.set_context(nick="x") is agent


def test_set_context_rejects_undeclared_name():
    """A typo'd name raises instead of being dropped."""
    agent = make_chatter(
        system_prompt="$nick",
        context_params=(ContextParam(name="nick"),),
    )
    with pytest.raises(PromptParameterError, match="nik"):
        agent.set_context(nik="Monica28")


def test_set_context_rejects_when_nothing_declared():
    """An agent declaring nothing accepts no context."""
    agent = make_chatter(system_prompt="hola")
    with pytest.raises(PromptParameterError, match="none"):
        agent.set_context(nick="Monica28")


def test_set_context_merges_across_calls():
    """Two calls accumulate rather than replace."""
    agent = make_chatter(
        system_prompt="$a $b",
        context_params=(
            ContextParam(name="a"),
            ContextParam(name="b"),
        ),
    )
    agent.set_context(a="1")
    agent.set_context(b="2")
    assert agent.context_values == {"a": "1", "b": "2"}


def test_set_context_coerces_to_string():
    """A non-string value is coerced, not rejected."""
    agent = make_chatter(
        system_prompt="$age",
        context_params=(ContextParam(name="age"),),
    )
    agent.set_context(age=28)
    assert agent.resolved_system_prompt == "28"


def test_context_values_is_a_copy():
    """Mutating the returned dict does not affect the agent."""
    agent = make_chatter(
        system_prompt="$nick",
        context_params=(ContextParam(name="nick"),),
    )
    agent.set_context(nick="x")
    agent.context_values["nick"] = "tampered"
    assert agent.resolved_system_prompt == "x"


def test_a_copy_keeps_the_supplied_context():
    """`with_overrides` must not hand back the defaults.

    The regression: a recorder copied the agent to attach a
    callback, and the copy answered with every default.
    """
    agent = make_chatter(
        system_prompt="Hablas con $nick.",
        context_params=(ContextParam(name="nick", default="ella"),),
    )
    agent.set_context(nick="Monica28")

    copy = agent.with_overrides(temperature=0.1)

    assert copy.resolved_system_prompt == "Hablas con Monica28."


def test_a_copy_drops_context_it_no_longer_declares():
    agent = make_chatter(
        system_prompt="$a $b",
        context_params=(ContextParam(name="a"), ContextParam(name="b")),
    )
    agent.set_context(a="1", b="2")

    copy = agent.with_overrides(context_params=(ContextParam(name="a"),))

    assert copy.context_values == {"a": "1"}


def test_a_copy_does_not_share_context_with_the_original():
    agent = make_chatter(
        system_prompt="$nick",
        context_params=(ContextParam(name="nick"),),
    )
    agent.set_context(nick="x")
    copy = agent.with_overrides(temperature=0.1)

    copy.set_context(nick="y")

    assert agent.resolved_system_prompt == "x"


# --- resolved_context -----------------------------------------


def test_resolved_context_covers_defaults_and_resolvers():
    """Not just what `set_context` supplied: what the model reads."""
    agent = make_chatter(
        system_prompt="$nick $moment $hint",
        context_params=(
            ContextParam(name="nick", default="desconocido"),
            ContextParam(name="moment", resolver=_resolver_value),
            ContextParam(name="hint"),
        ),
    )
    assert agent.resolved_context == {
        "nick": "desconocido",
        "moment": "resolved",
        "hint": "",
    }


def test_resolved_context_prefers_a_supplied_value():
    agent = make_chatter(
        system_prompt="$nick",
        context_params=(ContextParam(name="nick", default="x"),),
    )
    agent.set_context(nick="Marta_87")
    assert agent.resolved_context == {"nick": "Marta_87"}


def test_resolved_context_is_what_the_prompt_gets():
    """The property and the substitution cannot drift apart."""
    agent = make_chatter(
        system_prompt="$nick habla $moment",
        context_params=(
            ContextParam(name="nick", default="desconocido"),
            ContextParam(name="moment", resolver=_resolver_value),
        ),
    )
    resolved = agent.resolved_context
    assert agent.resolved_system_prompt == (
        f"{resolved['nick']} habla {resolved['moment']}"
    )


def test_resolved_context_is_empty_when_nothing_is_declared():
    agent = make_chatter(system_prompt="hola")
    assert agent.resolved_context == {}


# --- Declaration validation -----------------------------------


def test_duplicate_declaration_rejected():
    """The same name cannot be declared twice."""
    with pytest.raises(PromptParameterError, match="twice"):
        make_chatter(
            context_params=(
                ContextParam(name="nick"),
                ContextParam(name="nick"),
            ),
        )


def test_clash_with_prompt_params_rejected():
    """A name cannot be both fixed and per-conversation."""
    with pytest.raises(PromptParameterError, match="both"):
        make_chatter(
            prompt_params={"nick": "ella"},
            context_params=(ContextParam(name="nick"),),
        )


def test_non_context_param_entry_rejected():
    """Entries must be ContextParam instances."""
    with pytest.raises(PromptParameterError):
        make_chatter(context_params=({"name": "nick"},))


def test_list_is_normalised_to_a_tuple():
    """A list of declarations is accepted and frozen."""
    agent = make_chatter(context_params=[ContextParam(name="nick")])
    assert isinstance(agent.context_params, tuple)


# --- Clock helpers --------------------------------------------


@pytest.mark.parametrize(
    "hour,expected",
    [
        (3, "night"),
        (9, "morning"),
        (15, "afternoon"),
        (23, "evening"),
    ],
)
def test_time_of_day_buckets(hour, expected, monkeypatch):
    """The clock helper maps hours to parts of the day."""

    class FakeDateTime:
        @staticmethod
        def now():
            return type("T", (), {"hour": hour})()

    monkeypatch.setattr("agents7.agents.context.datetime", FakeDateTime)
    assert time_of_day() == expected


def test_spanish_time_of_day(monkeypatch):
    """The Spanish helper phrases the same buckets."""
    monkeypatch.setattr(
        "agents7.agents.context.time_of_day", lambda: "morning"
    )
    assert spanish_time_of_day() == "por la mañana"


@pytest.mark.parametrize(
    "index, expected",
    [
        (0, "lunes"),
        (1, "martes"),
        (2, "miércoles"),
        (3, "jueves"),
        (4, "viernes"),
        (5, "sábado"),
        (6, "domingo"),
    ],
)
def test_spanish_weekday(index, expected, monkeypatch):
    """Monday is 0, as `datetime.weekday()` counts it."""

    class FakeDateTime:
        @staticmethod
        def now():
            return type("T", (), {"weekday": lambda self: index})()

    monkeypatch.setattr("agents7.agents.context.datetime", FakeDateTime)
    assert spanish_weekday() == expected


# --- YAML loading ---------------------------------------------


def _write_config(tmp_path, context_params) -> str:
    config = {
        "name": "ctx-chatter",
        "version": 1,
        "engine": {"provider": "ollama", "model": "qwen3"},
        "system_prompt": "Hablas con $nick.",
        "context_params": context_params,
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return str(path)


def test_yaml_declares_default_and_description(tmp_path):
    """A mapping entry carries default and description."""
    path = _write_config(
        tmp_path,
        {"nick": {"default": "ella", "description": "Her nick"}},
    )
    agent = Chatter.from_yaml(path)
    assert agent.context_params[0].name == "nick"
    assert agent.context_params[0].default == "ella"
    assert agent.context_params[0].description == "Her nick"
    assert agent.resolved_system_prompt == "Hablas con ella."


def test_yaml_scalar_entry_is_the_default(tmp_path):
    """A scalar entry is read as the default value."""
    path = _write_config(tmp_path, {"nick": "ella"})
    agent = Chatter.from_yaml(path)
    assert agent.context_params[0].default == "ella"


def test_yaml_callable_is_imported(tmp_path):
    """A dotted path is imported and called per message."""
    path = _write_config(
        tmp_path,
        {"nick": {"callable": "test_context_params._resolver_value"}},
    )
    agent = Chatter.from_yaml(path)
    assert agent.resolved_system_prompt == "Hablas con resolved."


def test_yaml_unimportable_callable_raises(tmp_path):
    """A bad dotted path fails at load time, not at chat time."""
    path = _write_config(
        tmp_path, {"nick": {"callable": "nope.nothing_here"}}
    )
    with pytest.raises(AgentConfigurationError, match="could not be"):
        Chatter.from_yaml(path)


def test_yaml_non_mapping_section_raises(tmp_path):
    """context_params must be a mapping of name to declaration."""
    path = _write_config(tmp_path, ["nick"])
    with pytest.raises(AgentConfigurationError, match="mapping"):
        Chatter.from_yaml(path)


def test_yaml_absent_section_leaves_none(tmp_path):
    """A config without the section declares nothing."""
    config = {
        "name": "plain",
        "version": 1,
        "engine": {"provider": "ollama", "model": "qwen3"},
        "system_prompt": "Hola.",
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    agent = Chatter.from_yaml(str(path))
    assert agent.context_params is None


# --- Run identity ---------------------------------------------
#
# A run's fingerprint hashes `system_prompt_identity`, so anything
# resolved into it splits one configuration's measurement history
# into a band per resolved value — a clock resolver into one band
# per part of the day. See `.claude/rules/measurement-needs-
# history.md`, rule 2.

_CLOCK = {"now": "por la mañana"}


def _clock() -> str:
    """Resolver reading a clock the test moves."""
    return _CLOCK["now"]


def _timed_chatter(**kwargs) -> Chatter:
    return make_chatter(
        system_prompt="Es $moment, y ella es $nick.",
        context_params=(
            ContextParam(name="moment", resolver=_clock),
            ContextParam(name="nick", default="ella"),
        ),
        **kwargs,
    )


def test_a_resolver_does_not_move_the_identity():
    """The regression: the clock must not reach the fingerprint."""
    _CLOCK["now"] = "por la mañana"
    morning = _timed_chatter().system_prompt_identity
    _CLOCK["now"] = "por la noche"
    night = _timed_chatter().system_prompt_identity
    assert morning == night


def test_runs_at_different_hours_share_a_fingerprint():
    """What that identity is for, at the level it is consumed."""
    contract = StyleContract(
        rules=(),
        scripts=(ConversationScript(id="cold", lines=("bien",)),),
    )
    # Hashed while each clock reads, not both at the end: the
    # identity resolves when it is asked for, so building both
    # agents first would read one clock twice and pass either way.
    _CLOCK["now"] = "por la mañana"
    morning = fingerprint_run(contract, _timed_chatter())
    _CLOCK["now"] = "de madrugada"
    night = fingerprint_run(contract, _timed_chatter())
    assert morning == night


def test_a_supplied_value_does_not_move_the_identity():
    """`set_context` is input, not configuration."""
    plain = _timed_chatter()
    supplied = _timed_chatter().set_context(nick="Monica28")
    assert plain.system_prompt_identity == supplied.system_prompt_identity


def test_the_identity_declares_every_param():
    """Left as placeholders, each declared after the prompt."""
    identity = _timed_chatter().system_prompt_identity
    assert "$moment" in identity
    assert "$nick" in identity
    assert "context|nick|ella|" in identity
    assert f"context|moment||{__name__}._clock" in identity


def test_changing_a_default_moves_the_identity():
    """The declaration still discriminates: a default is config."""
    before = _timed_chatter().system_prompt_identity
    after = make_chatter(
        system_prompt="Es $moment, y ella es $nick.",
        context_params=(
            ContextParam(name="moment", resolver=_clock),
            ContextParam(name="nick", default="desconocida"),
        ),
    ).system_prompt_identity
    assert before != after


def test_changing_the_resolver_moves_the_identity():
    """So does swapping where a value comes from."""
    before = _timed_chatter().system_prompt_identity
    after = make_chatter(
        system_prompt="Es $moment, y ella es $nick.",
        context_params=(
            ContextParam(name="moment", resolver=_resolver_value),
            ContextParam(name="nick", default="ella"),
        ),
    ).system_prompt_identity
    assert before != after


def test_the_resolved_prompt_still_fills_them():
    """The identity is neutralised; the real prompt is not."""
    _CLOCK["now"] = "por la noche"
    agent = _timed_chatter().set_context(nick="Monica28")
    assert (
        agent.resolved_system_prompt == "Es por la noche, y ella es Monica28."
    )


# --- Derived params -------------------------------------------


def _shout(value: str) -> str:
    """Module-level one-argument resolver, importable by path."""
    return value.upper()


def _derived_chatter():
    return make_chatter(
        system_prompt="$nick / $hint",
        context_params=(
            ContextParam(name="hint", resolver=_shout, derived_from="nick"),
            ContextParam(name="nick", default="ella"),
        ),
    )


def test_derived_param_reads_its_source_default():
    """Declared before its source, and still computed after it."""
    assert _derived_chatter().resolved_context == {
        "nick": "ella",
        "hint": "ELLA",
    }


def test_derived_param_follows_a_supplied_source():
    """Supplying the source is enough; the derived one comes along."""
    agent = _derived_chatter().set_context(nick="Monica28")
    assert agent.resolved_system_prompt == "Monica28 / MONICA28"


def test_derived_param_tracks_a_later_source_change():
    """No stale value from the previous conversation."""
    agent = _derived_chatter().set_context(nick="Monica28")
    agent.set_context(nick="Ana")
    assert agent.resolved_context["hint"] == "ANA"


def test_supplied_value_wins_over_a_derived_one():
    agent = _derived_chatter().set_context(nick="Monica28", hint="mine")
    assert agent.resolved_context == {"nick": "Monica28", "hint": "mine"}


def test_derived_param_reads_a_resolved_source():
    """The source may itself come from a zero-argument resolver."""
    agent = make_chatter(
        system_prompt="$hint",
        context_params=(
            ContextParam(name="moment", resolver=_resolver_value),
            ContextParam(name="hint", resolver=_shout, derived_from="moment"),
        ),
    )
    assert agent.resolved_system_prompt == "RESOLVED"


def test_derived_resolver_failure_falls_back():
    """A resolver that raises on the value yields the default."""

    def _bad(value: str) -> str:
        raise ValueError(value)

    agent = make_chatter(
        system_prompt="$hint",
        context_params=(
            ContextParam(name="nick", default="ella"),
            ContextParam(
                name="hint",
                default="nada",
                resolver=_bad,
                derived_from="nick",
            ),
        ),
    )
    assert agent.resolved_system_prompt == "nada"


def test_derived_param_needs_a_resolver():
    with pytest.raises(PromptParameterError, match="no callable"):
        ContextParam(name="hint", derived_from="nick")


def test_derived_param_cannot_read_itself():
    with pytest.raises(PromptParameterError, match="from itself"):
        ContextParam(name="hint", resolver=_shout, derived_from="hint")


def test_derived_param_needs_a_declared_source():
    with pytest.raises(PromptParameterError, match="not declared"):
        make_chatter(
            system_prompt="$hint",
            context_params=(
                ContextParam(
                    name="hint", resolver=_shout, derived_from="nick"
                ),
            ),
        )


def test_derived_param_cannot_read_a_derived_one():
    with pytest.raises(PromptParameterError, match="itself derived"):
        make_chatter(
            system_prompt="$b",
            context_params=(
                ContextParam(name="nick"),
                ContextParam(name="a", resolver=_shout, derived_from="nick"),
                ContextParam(name="b", resolver=_shout, derived_from="a"),
            ),
        )


def test_yaml_from_wires_a_derived_param(tmp_path):
    """`from` names the source; the callable takes its value."""
    path = _write_config(
        tmp_path,
        {
            "nick": {"default": "ella"},
            "hint": {
                "callable": "test_context_params._shout",
                "from": "nick",
            },
        },
    )
    agent = Chatter.from_yaml(path).set_context(nick="Monica28")
    assert agent.resolved_context["hint"] == "MONICA28"


def test_the_identity_names_what_a_param_is_derived_from():
    """Deriving a value is configuration, so it discriminates."""
    identity = _derived_chatter().system_prompt_identity
    assert f"context|hint||{__name__}._shout|from:nick" in identity
    assert "context|nick|ella|\n" in identity + "\n"


def test_a_derived_value_does_not_move_the_identity():
    plain = _derived_chatter()
    supplied = _derived_chatter().set_context(nick="Monica28")
    assert plain.system_prompt_identity == supplied.system_prompt_identity
