# Gated DeltaNet chunk_fwd_h — Manual Mutation Iteration 01

## Baseline
Target: `helion_targets/gated_deltanet_chunk_fwd_h/seed.py`

Known baseline structure:
- uniform `block_sizes=[64]`
- uniform `num_warps=4`
- uniform `num_stages=1`
- sequential chunk recurrence over `T` with `chunk_size=64`

Primary goal:
- lower `mean_runtime_us`

## Step 2 — 10 hypotheses
1. Increase `num_warps` and `num_stages` for benchmark shapes to better pipeline the chunk recurrence.
2. Use `tensor_descriptor` indexing on all shapes.
3. Use `tensor_descriptor` only on the larger benchmark shapes.
4. Pre-chunk inputs explicitly and iterate over chunk index instead of strided sequence tiles.
5. Apply `chunk_fwd_h` ACF only on the large benchmark shapes.
6. Deepen only the 1024-shape pipeline while keeping the rest near baseline.
7. Use a stronger config for the `V=128` test shape to avoid underfitting that path.
8. Try `block_ptr` indexing on benchmark shapes.
9. Increase V-tiling aggressiveness together with higher warps.
10. Combine explicit chunk layout with moderate warp/stage tuning.

## Step 3 — Discard 5 on theory
Discarded:
- **H2** — global descriptor indexing is too broad for round 1.
- **H8** — block pointer is weaker and less directly motivated than large-shape descriptor.
- **H9** — larger V-tiling plus higher warps mixes too many effects.
- **H7** — test-shape-only optimization is weakly aligned with the runtime objective.
- **H10** — explicit chunk layout plus schedule tuning is better split into cleaner children.

Retained:
- **H1** conservative benchmark warp/stage tuning
- **H3** large-shape descriptor only
- **H4** explicit chunk layout
- **H5** large-shape-only ACF
- **H6** deepest pipeline only on the largest benchmark

## Step 4 — Set-level gaps
Remaining gaps:
- no pure exploit child that stays close to baseline
- no structure-only child that isolates chunk layout benefits

## Step 5 — Two extra hypotheses
11. Pure exploit: keep baseline everywhere except modestly raise benchmark warp/stage settings.
12. Structure-only: explicitly chunk `k`, `w`, `u`, and `g`, but keep configs near baseline.

## Distilled first-round ideas
- `m01_conservative_warp_stage.py`
- `m02_deeper_pipeline.py`
- `m03_explicit_chunk_layout.py`
- `m04_large_shape_descriptor.py`
- `m05_large_shape_acf.py`

## Evaluation plan
- run baseline seed
- run all five variants
- compare validity first, runtime second
- use round-1 results to decide whether the winning direction is schedule tuning or structural chunk layout
