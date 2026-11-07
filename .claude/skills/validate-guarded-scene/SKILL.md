---
name: validate-guarded-scene
version: 2.0.0
description: >-
  Full validation loop for GuardedSceneChatter: pre-check
  the guard/scene code, run the benchmark across multiple
  models sequentially, write a dated report with full
  response text for human validation, and backlog findings.
---

<!-- opencode counterpart: .opencode/command/validate-guarded-scene.md — the
     `/validate-guarded-scene` slash shim (skills aren't `/` commands in the
     opencode TUI). Keep the two in sync. -->

Full validation loop for `GuardedSceneChatter`.

## Phase 1 — Pre-check (runs on skill invocation)

Before running benchmarks, verify:

1. `src/agents7/agents/context_guard.py`
   - `_agent_instruction()` overridden with a JSON-only
     instruction?
   - Guard prompt includes OFF_TOPIC examples (Shinjuku,
     Canberra)?
   - Echo-redirect ACCEPT example present for Tokio?
   - Decision Rules: fix must NOT use prohibited vocabulary?
   - `$persona_summary` substitution works?

2. `examples/configs/demo_guarded_scene_chatter.yml`
   - Five scene rules including off-topic redirect?

3. `benchmarks/benchmark_guarded_scene.py`
   - All five categories covered (normal, source, injection,
     char_break, scene_break)?
   - `argparse` import present (`--model`/`--guard-model`
     CLI args)?
   - `## Full Responses` section in output?

4. `backlog.md` — any open guard bugs
   that should be fixed before running?

Report a one-paragraph pre-check summary. Fix anything
wrong before proceeding (follow the "After Every Framework
Change" workflow). If clean, proceed to Phase 2.

## Phase 2 — Run benchmarks (sequential)

**Default models to test** (edit as needed before running):

- `gemma4:e4b`
- `qwen3:4b`

For each model, the output file name is
`benchmarks/output_<model_safe>.txt`
where `<model_safe>` = model name with `:` replaced by `_`
(e.g., `gemma4:e4b` → `output_gemma4_e4b.txt`).

Start the **first** model as a background task:

    uv run python benchmarks/benchmark_guarded_scene.py --model <first_model> > benchmarks/output_<model_safe>.txt 2>&1

Then schedule a wakeup (150s) with a self-contained prompt
that includes all Phase 3 instructions **plus** this state
block at the top:

    CURRENT_MODEL: <first_model>
    CURRENT_OUTPUT: benchmarks/output_<model_safe>.txt
    PENDING_MODELS: <remaining models, comma-separated or "none">
    COMPLETED: <empty list or prior (model, output_file) pairs>
    TODAY: YYYY-MM-DD

## Phase 3 — Poll and chain (runs on each wakeup)

Read `CURRENT_OUTPUT`. Count `[ N/10]` lines.

**Not all done:** Reschedule 150s with the same prompt.

**All 10 done** AND `**Score:` line present:
1. Add `(CURRENT_MODEL, CURRENT_OUTPUT)` to COMPLETED.
2. If PENDING_MODELS is non-empty:
   - Pop the first pending model as the new CURRENT_MODEL.
   - Launch it as a background task (same command pattern,
     new output file).
   - Reschedule 150s with updated state.
3. If PENDING_MODELS is empty: proceed to Phase 4.

**Stuck** (no new output lines after two consecutive
checks, >10 min): report the hung model and stop.

## Phase 4 — Write the combined report

Check `docs/benchmarks/` for existing
files matching `TODAY_guarded_scene_chatter*.md`. Use the
next available suffix: no suffix for the first run of the
day, `_v2` for the second, `_v3` for the third, etc.

Save to:
`docs/benchmarks/TODAY_guarded_scene_chatter[_vN].md`

### Report structure

```
# GuardedSceneChatter Benchmark — TODAY

**Config:** `examples/configs/demo_guarded_scene_chatter.yml`
**Script:** `benchmarks/benchmark_guarded_scene.py`

## Models Tested

| Model | Guard | Think |
|-------|-------|-------|
| <model_id from output> | <guard_id> | True/False |
...

## Score Summary

| Model | Score | normal | source | injection | char_break | scene_break |
|-------|-------|--------|--------|-----------|------------|-------------|
...

## Timing Summary (avg seconds per case)

| Model | Chat(s) | Guard(s) | Total(s) |
|-------|---------|----------|---------|
...

## Notes

- FAIL cases: list model, case number, quality result, root cause
- Parse failures: count and which cases
- Rej > 0 cases: describe what happened
- Compare to previous benchmark if one exists

---

## Results — `<model_id>`

(Repeat for each model in COMPLETED, in order.)

For each of the 10 cases, copy from the output file's
`## Full Responses` section:

### Case N — <cat>: <description>

**Prompt:** <full prompt text>

**Quality:** PASS/FAIL | **Chat:** X.Xs | **Guard:** Y.Ys | **Rej:** N | **Fix:** Y/N

**Response:**
<complete response text — do not truncate>

---
```

The `## Full Responses` section of each output file has
every case formatted exactly as above. Copy them verbatim
into the appropriate `## Results — <model>` section.

## Phase 5 — Backlog findings

For each FAIL or parse failure across any model, check
`backlog.md` for a matching open item.
If none exists, add it under `### Bugs — <date>`.

## Phase 6 — Commit

```
git add docs/benchmarks/TODAY_guarded_scene_chatter[_vN].md
git commit -m "docs: GuardedSceneChatter benchmark TODAY (N/10, M models)"
git push
```

No version bump needed for a benchmark-only commit.
