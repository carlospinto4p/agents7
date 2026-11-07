# Agent packages

An agent can live in a repository of its own, as an installable Python
package. [agent-sentiment_classifier](https://github.com/carlospinto4p/agent-sentiment_classifier)
is a complete example.

## Depending on agents7

agents7 is not on PyPI. Depend on a tagged release by its git URL, in
`dependencies`:

```toml
dependencies = [
    "agents7[openai] @ git+https://github.com/carlospinto4p/agents7@v9.25.9",
]

[tool.hatch.metadata]
allow-direct-references = true
```

hatchling refuses to build a package with a git URL dependency unless
`allow-direct-references` is set.

Put the URL in `dependencies`, not only in `[tool.uv.sources]`. With a
bare name, `pip` looks for `agents7` on PyPI and finds nothing, or a
different package.

## One tag per environment

A git dependency names one exact tag. It cannot say "9.21 or later". So
every package installed into one environment must name the same agents7
tag. If agents7-ui pins `v9.25.9` and an agent pins a later tag,
installing both fails.

## Trying an unreleased agents7 change

Install a local checkout over the pinned tag:

```bash
uv pip install -e <agents7 checkout>
```

The next `uv sync` puts the tag back.

## Showing the agent in agents7-ui

Declare the agent's config in the `agents7.agents` entry point group:

```toml
[project.entry-points."agents7.agents"]
sentiment_classifier = "sentiment_classifier:CONFIG_PATH"
```

The value names a module attribute holding the path to the agent's
`config.yml`. Install the package into the UI backend's environment and
the UI lists it. An eval set at `data/golden_set.jsonl` beside the
config is found as well.
