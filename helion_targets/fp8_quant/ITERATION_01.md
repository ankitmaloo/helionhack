# FP8 Quant — Manual Mutation Iteration 01

## Baseline
Target: `helion_targets/fp8_quant/seed.py`

Known baseline metrics from prior evaluation:
- `mean_runtime_us`: `32.16666666666667`
- `min_runtime_us`: `7.3`
- `correctness`: `1.0`

Primary goal:
- lower `mean_runtime_us`

## Step 2 — 10 hypotheses
1. Increase `block_sizes` and `num_stages` for benchmark shapes to improve occupancy and memory pipelining.
2. Switch to `indexing="tensor_descriptor"` for the whole kernel to improve aligned memory movement on B200.
3. Use `pid_type="persistent_interleaved"` for large shapes to reduce scheduling overhead.
4. Add `advanced_controls_file` for benchmark shapes to let PTXAS use B200-tuned scheduling.
5. Use `reduction_loops=[64]` on `group_size=128` shapes to reduce register pressure.
6. Replace `row / scale[:, None]` with reciprocal multiply to reduce division cost.
7. Introduce `hl.register` staging for the row tile before reduction and writeback.
8. Split configs aggressively by small vs large shapes so test shapes remain conservative while benchmark shapes become more aggressive.
9. Use `num_sm_multiplier` with persistent scheduling to increase multi-occupancy.
10. Fuse clamp and writeback logic into a more explicit temporary pipeline to encourage better codegen.

## Step 3 — Discard 5 hypotheses on theory
Discarded:
- **H7** — register staging is higher risk and less attributable than plain config tuning.
- **H10** — explicit temporary pipeline is too vague and likely to increase code size without a clear bottleneck win.
- **H9** — `num_sm_multiplier` is interesting, but it adds another variable on top of persistent scheduling and should not be mixed into round 1.
- **H5** — reduction looping is useful if register pressure is the dominant issue, but the first round should test simpler schedule changes first.
- **H8** — the seed already has per-shape configs; further aggressive split logic is better treated as an implementation pattern than a standalone hypothesis.

Retained from theory:
- **H1** block size + staging
- **H2** tensor descriptor indexing
- **H3** persistent scheduling
- **H4** ACF-based compiler guidance
- **H6** reciprocal multiply

## Step 4 — Set-level gaps
The retained set still misses:
- a conservative control mutation with minimal semantic change
- a mutation that combines a safe math rewrite with moderate config tuning
- a mutation that tests whether the performance ceiling is compiler-guided rather than schedule-guided

## Step 5 — Two extra hypotheses
11. Conservative control: keep kernel math unchanged, only raise `num_stages` and moderate benchmark `block_sizes`.
12. Mixed math + schedule: use reciprocal multiply while keeping configs moderate and compatible.

## Distilled first-round ideas
### Main idea A
Conservative occupancy tuning.

### Main idea B
Tensor descriptor indexing with moderate staging.

### Main idea C
Persistent scheduling on large shapes.

### Main idea D
ACF-guided benchmark configuration.

## Manual mutations to implement
Five tracked files will be created:
- `m01_blocksize_staging.py`
- `m02_tensor_descriptor.py`
- `m03_persistent_interleaved.py`
- `m04_acf_tuned.py`
- `m05_reciprocal_scale.py`

## Mapping from hypotheses to files
- `m01_blocksize_staging.py` tests H1 + H11.
- `m02_tensor_descriptor.py` tests H2.
- `m03_persistent_interleaved.py` tests H3.
- `m04_acf_tuned.py` tests H4.
- `m05_reciprocal_scale.py` tests H6 + H12.

## Evaluation plan
- run baseline seed as control
- run all five mutations
- isolate every variant in its own copied problem directory so parallel runs are safe
- compare validity first, runtime second

## Round-1 success criteria
A mutation is interesting if it:
- passes correctness
- improves `mean_runtime_us`
- or reveals a strong failure mode that narrows the search space for round 2
