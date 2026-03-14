# Manual Kernel Mutation Algorithm

## Goal
Create, track, and evaluate manual kernel mutations without relying on AlphaEvolve's LLM mutation step.

## Inputs
- A target folder with `task.json`, `seed.py`, and context files.
- A baseline evaluation path that can run correctness and benchmark checks.
- A mutation budget for one or more rounds.

## Outputs
- A `skill.md` file describing the code layer being improved.
- A per-iteration hypothesis file.
- Separate mutation source files.
- Run artifacts with per-variant metrics and failure reasons.
- A second-round refinement based on failed or weak first-round variants.

## Procedure

### Step 0 — Write the algorithm down
Create this file first so the mutation loop is explicit and repeatable.

### Step 1 — Identify the code layer to improve
Choose one dominant layer before editing code.

Examples:
- Kernel schedule/config layer
- Memory movement layer
- Math expression layer
- Launch geometry layer

Do not optimize multiple unrelated layers at once in the first round unless a mutation is deliberately cross-layer.

### Step 2 — Create `skill.md`
Write a target-specific `skill.md` that captures:
- objective
- invariants that must not change
- safe mutation surface
- high-risk edits to avoid
- likely optimization levers
- measurement protocol
- baseline metrics

This becomes the fixed context for the next mutations.

### Step 3 — Generate 10 hypotheses
Cover the solution space broadly.

Good hypothesis classes:
- occupancy and warp tuning
- block size changes
- staging and software pipelining
- indexing mode changes
- persistent scheduling
- compiler hint or ACF usage
- register-pressure relief
- algebraic simplification
- writeback pattern changes
- shape-specialized policies

### Step 4 — Discard 5 hypotheses on theory alone
Remove hypotheses that are likely to fail for one of these reasons:
- too invasive
- unclear correctness story
- compile risk too high
- overlaps too much with a stronger hypothesis
- mismatched with the kernel bottleneck

### Step 5 — Identify set-level gaps
Ask what the current remaining set is missing.

Typical gaps:
- no safe control mutation
- no compiler-directed mutation
- no light algorithmic mutation
- no mutation targeting small-shape behavior
- no mutation targeting large-shape behavior

### Step 6 — Add 2 gap-filling hypotheses
Add two hypotheses specifically to cover the missing space found in Step 5.

### Step 7 — Distill to 4 main ideas
Choose four ideas to run in the first batch.
Selection criteria:
- one conservative control
- one strong config mutation
- one scheduling/indexing mutation
- one compiler or math mutation

A fifth mutation can still be carried as an extra exploratory candidate if budget allows.

### Step 8 — Implement each mutation as a separate file
Do not overwrite the seed.
Create one file per mutation with a stable name and short intent label.

Recommended naming:
- `manual_mutations/round1/m01_<idea>.py`
- `manual_mutations/round1/m02_<idea>.py`
- ...

### Step 9 — Run baseline plus mutations
Evaluate:
- baseline seed
- all selected mutations

If evaluations run in parallel, isolate each variant so runs do not share the same mutable submission path.

### Step 10 — Track outcomes per mutation
Record for each mutation:
- validity
- correctness pass/fail
- mean runtime
- min runtime
- compile/test/benchmark time
- failure reasons
- whether the idea improved the baseline

### Step 11 — Analyze what did not work
For each weak mutation, classify the failure:
- compile failure
- correctness failure
- performance regression
- benchmark variance / unstable gain

Then write a short hypothesis for why it failed.

### Step 12 — Run a second iteration
Repeat the same structure using first-round knowledge:
- update the surviving hypothesis set
- discard ideas contradicted by results
- target the newly revealed bottleneck
- implement fewer, sharper mutations

## Tracking Rules
- Keep every mutation in a separate file.
- Never lose the baseline.
- Prefer small, attributable edits over mixed large rewrites.
- Parallelism is allowed only when artifact isolation is guaranteed.
- Treat failed mutations as signal, not waste.

## Round Exit Criteria
A round is complete when:
- all chosen mutations were evaluated
- results are saved
- failed ideas have a written explanation
- the next round uses that explanation explicitly
