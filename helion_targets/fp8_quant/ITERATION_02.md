# FP8 Quant — Manual Mutation Iteration 02

## Round-1 result summary
Baseline remained best.

Measured results:
- `baseline_seed`: `32.1667 us`
- `m01_blocksize_staging`: `36.4 us`
- `m02_tensor_descriptor`: `44.7333 us`
- `m03_persistent_interleaved`: `39.0333 us`
- `m04_acf_tuned`: `44.9333 us`
- `m05_reciprocal_scale`: `63.0667 us`

## What did not work
- Larger block sizes plus higher staging regressed the kernel.
- Global `tensor_descriptor` indexing regressed the kernel.
- ACF plus tensor descriptor regressed even more.
- Reciprocal multiply caused the worst regression and high variance on the largest benchmark.
- Persistent scheduling improved the smallest benchmark but badly hurt the largest benchmark.

## Hypotheses for why round 1 failed
1. The seed schedule is already close to the best occupancy/register-pressure point.
2. This kernel is too simple and bandwidth-bound to benefit from more aggressive global schedule changes.
3. `tensor_descriptor` adds setup overhead without enough reuse to amortize it.
4. The tested ACF files are mismatched to the chosen schedule or access pattern.
5. The reciprocal rewrite generates worse code for the large shape or increases sensitivity to compiler lowering.
6. Persistent scheduling is only helpful for smaller-N workloads and should not be applied to the largest benchmark shape.

## Step 2 — 10 new hypotheses
1. Keep baseline everywhere except use the successful persistent schedule only on the smaller benchmark shapes.
2. Keep baseline schedule but add `reduction_loops=[64]` on benchmark shapes to relieve register pressure at `group_size=128`.
3. Keep baseline schedule but change benchmark shapes to `indexing="block_ptr"` instead of `tensor_descriptor`.
4. Keep baseline schedule but add `l2_groupings=[8]` for benchmark shapes.
5. Keep baseline schedule but use persistent scheduling only on the very first benchmark shape.
6. Keep baseline schedule but raise `num_warps` only for the middle benchmark shape.
7. Keep baseline schedule but lower block size only for the first benchmark shape.
8. Keep baseline schedule but use `reduction_loops=[32]` only on the largest benchmark shape.
9. Keep baseline schedule but pair `block_ptr` with `l2_groupings=[8]` on benchmark shapes.
10. Keep baseline schedule but use a hybrid persistent configuration only for the first two benchmark shapes and baseline for the largest.

## Step 3 — Discard 5 on theory
Discarded:
- **H6** — isolated warp increase is weakly motivated after block/staging regressions.
- **H7** — smaller block size for the first benchmark shape is too speculative without stronger evidence.
- **H8** — `reduction_loops=[32]` is more invasive and less standard than `[64]` for `group_size=128`.
- **H9** — combining `block_ptr` and `l2_groupings` mixes two effects and hurts attribution.
- **H5** — persistent only on the first benchmark shape is dominated by the broader hybrid hypothesis.

Retained:
- **H1** smaller-benchmark persistent hybrid
- **H2** conservative reduction loop
- **H3** conservative block pointer indexing
- **H4** conservative L2 grouping
- **H10** hybrid persistent on first two benchmark shapes with baseline on the largest

## Step 4 — Set-level gaps
Remaining gaps:
- no control mutation that changes only large-shape behavior
- no mutation that preserves baseline schedule and only changes dataflow hints
- no mutation specifically targeting the observation that large-shape regression dominates the mean

## Step 5 — Two extra hypotheses
11. Use the hybrid persistent schedule for the first two benchmark shapes but keep the exact baseline configuration for the largest shape.
12. Keep baseline everywhere except add a cache-locality hint on benchmark shapes with no other schedule changes.

## Distilled main ideas
1. **Hybrid persistent** — apply persistent scheduling only where round 1 hinted at benefit.
2. **Block pointer indexing** — test a lighter indexing change than tensor descriptors.
3. **Reduction loop relief** — test a small register-pressure intervention.
4. **L2 locality hint** — test cache locality without changing occupancy.

## Round-2 mutation files
- `m01_hybrid_persistent.py`
- `m02_block_ptr_benchmark.py`
- `m03_reduction_loops_benchmark.py`
- `m04_l2_grouping_benchmark.py`

## Evaluation objective
A round-2 mutation is promising if it:
- remains valid
- stays within the baseline correctness envelope
- beats `32.1667 us`
- or shows a more interpretable shape-specific tradeoff than round 1
