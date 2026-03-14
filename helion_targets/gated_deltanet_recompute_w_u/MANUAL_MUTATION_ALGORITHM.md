# Manual Mutation Algorithm — `gated_deltanet_recompute_w_u`

## Goal
Run Codex-driven kernel mutation for `gated_deltanet_recompute_w_u` without using AlphaEvolve's mutator, while preserving the same disciplined search loop:
- define the target layer
- generate and prune hypotheses
- implement isolated children
- evaluate them in parallel on the B200 SSH substrate
- use failures as signal for the next round

## Baseline
- Baseline source: `seed.py`
- Problem directory on remote substrate: `gated_deltanet_recompute_w_u_py`
- Evaluation path: `manual_mutation_runner.py` over the SSH substrate in `../substrate.json`

## Round Algorithm

### Step 0 — Freeze the procedure
Create this file first so the search loop is explicit and repeatable.

### Step 1 — Pick one dominant code layer
For round 1, mutate the chunk-matmul execution layer:
- outer loop structure across chunks
- Helion config per shape
- indexing / scheduling policy
- compiler guidance via ACF

Do not mix in a new mathematical objective or API change.

### Step 2 — Write `skill.md`
Capture:
- what the kernel computes
- invariants that must hold
- safe mutation surface
- dangerous edits to avoid
- likely performance levers
- how results will be measured

This file is the stable context for all later children.

### Step 3 — Generate 10 hypotheses
Cover distinct parts of the solution space:
- conservative config tuning
- tensor descriptor indexing
- persistent scheduling
- explicit chunk parallelization
- ACF-guided compilation
- dual-output fusion
- host-side chunk reshape
- K/V-specific config split
- loop flattening
- register-pressure relief

### Step 4 — Discard 5 hypotheses on theory
Remove ideas that are too invasive, weakly justified, or overlapping with stronger ones.

### Step 5 — Identify set-level gaps
Check whether the retained set is missing:
- a conservative control
- a compiler-guided candidate
- a structural chunk-parallel candidate

### Step 6 — Add 2 gap-filling hypotheses
Add only the hypotheses that cover the missing space from Step 5.

### Step 7 — Distill to 4 main ideas
Pick four main ideas and optionally keep a fifth exploratory child:
1. conservative config control
2. indexing change
3. scheduler / persistent policy
4. explicit chunk-parallel structure
5. optional compiler-guided ACF child

### Step 8 — Implement children in separate files
Write one file per mutation under:
- `manual_mutations/round1/`

Never overwrite `seed.py`.

### Step 9 — Run baseline plus children in parallel over SSH
Use isolated remote problem copies so variants do not collide:

```bash
python3 manual_mutation_runner.py \
  --problem gated_deltanet_recompute_w_u \
  --variant-dir gated_deltanet_recompute_w_u/manual_mutations/round1 \
  --include-baseline \
  --parallelism 6 \
  --mode both \
  --run-name round1
```

### Step 10 — Record outcomes
For each child, capture:
- validity
- correctness pass/fail
- mean runtime
- min runtime
- compile/test/benchmark time
- failure reasons

### Step 11 — Explain what failed
For each non-winning child, classify the result:
- compile failure
- correctness failure
- runtime regression
- unstable or inconclusive benchmark

Write one short explanation for why the hypothesis likely failed.

### Step 12 — Repeat with narrowed hypotheses
Iteration 2 must explicitly use iteration-1 evidence:
- keep only ideas with signal
- reject contradicted theories
- add one or two new hypotheses targeted at the newly exposed bottleneck

## Operating Rules
- Keep mutations attributable: one dominant idea per file.
- Keep the seed available as the control.
- Use remote isolation for every child.
- Prefer evidence over intuition once runtime data exists.
