# Gated DeltaNet recompute_w_u — Manual Mutation Iteration 02

## Round-1 evidence
- Baseline: `49.666666666666664 us`
- Best child: `m01_conservative_warp_stage.py` at `42.86666666666667 us`
- `m02_tensor_descriptor.py` matched the winner within noise
- `m05_acf_descriptor.py` improved the large shapes but hurt the small benchmark shape
- Both `flatten_loops` candidates failed at config normalization before correctness

## What did not work, and why
- `flatten_loops=True` did not work because this kernel's current Helion config spec exposes zero flatten-loop entries. That is a tooling constraint, not a mathematical failure.
- The ACF-heavy child likely overfit the larger shapes and destabilized the smallest benchmark shape.
- Full descriptor indexing did not beat the simpler warp/stage tuning, which suggests schedule pressure is the dominant signal so far.

## Step 2 — 10 updated hypotheses
1. Keep the seed body and raise only `num_stages` further on the largest benchmark shape.
2. Keep the seed body and split K-path versus V-path configs more aggressively for the test `V=128` shape and the largest benchmark.
3. Re-test explicit chunk-axis restructuring without `flatten_loops`, since the earlier failure came from config normalization rather than the structure itself.
4. Use descriptor indexing only on the large benchmark shapes, not globally.
5. Apply ACF only to the large benchmark shapes and leave the small benchmark on the winning non-ACF config.
6. Use ACF only on the V-path, since V is the wider path on the test surface.
7. Raise `num_warps` to 16 on the largest benchmark shape to test whether more warp-level parallelism helps.
8. Lower `num_stages` back to 2 on the 512 benchmark while keeping 4 on the 1024 benchmark.
9. Reintroduce chunk-axis restructuring and pair it with stronger stage tuning.
10. Combine descriptor indexing with the current winning warp/stage schedule only for the largest benchmark shape.

## Step 3 — Discard 5 hypotheses on theory
Discarded:
- **H7** — `num_warps=16` is too aggressive for round 2 and risks wasting a slot on occupancy collapse.
- **H8** — stage rollback on the 512 shape is already close to the current winner and is too incremental to justify a full slot alone.
- **H9** — chunk-axis plus stronger stage tuning mixes two unverified ideas at once.
- **H2** — aggressive K/V-specific divergence is still better treated as an implementation detail within stronger schedule candidates.
- **H10** — descriptor-only on the largest shape overlaps too much with the stronger large-shape descriptor hypothesis.

Retained from theory:
- **H1** deeper pipeline on the largest benchmark
- **H3** explicit chunk axis without invalid flattening
- **H4** descriptor indexing only where it plausibly matters
- **H5** ACF only on large shapes
- **H6** ACF only on the V-path

## Step 4 — Set-level gaps
The retained set still misses:
- a pure exploit child that stays very close to the round-1 winner
- a structural child that cleanly isolates the chunk-axis rewrite

## Step 5 — Two extra hypotheses
11. Pure exploit: keep the winning round-1 schedule and deepen only the 1024-shape pipeline.
12. Structural retry: rerun the chunk-axis rewrite with seed-like configs and no unsupported flattening knobs.

## Distilled round-2 ideas
### Main idea A
Deeper pipeline on the winning schedule.

### Main idea B
Large-shape descriptor path only.

### Main idea C
Explicit chunk axis without `flatten_loops`.

### Main idea D
Large-shape-only ACF.

## Manual mutations to implement
Four tracked files:
- `m01_deeper_pipeline.py`
- `m02_large_shape_descriptor.py`
- `m03_explicit_chunk_axis.py`
- `m04_large_shape_acf.py`

## Evaluation plan
- run the baseline seed again as control
- run the four round-2 children in parallel over SSH
- treat any chunk-axis compile or correctness result as high-value evidence because round 1 never tested that structure cleanly

## Round-2 outcome
- `m03_explicit_chunk_axis.py` won decisively at `27.866666666666664 us`
- `m01_deeper_pipeline.py` and `m02_large_shape_descriptor.py` both landed at `42.2 us`, which is better than round 1 but far behind the chunk-axis rewrite
- `m04_large_shape_acf.py` improved over baseline but did not beat the non-ACF schedule variants

Per-shape picture for the winner:
- benchmark shape `(1, 64, 1, 64, 64)`: `17.1 us`
- benchmark shape `(2, 512, 3, 64, 64)`: `32.0 us`
- benchmark shape `(2, 1024, 3, 64, 64)`: `34.5 us`

## Why the non-winners likely did not win
- `m01_deeper_pipeline.py`: pipeline depth helped the 512-shape benchmark, but it did not address the core inefficiency of serial chunk traversal inside each program.
- `m02_large_shape_descriptor.py`: descriptor indexing remained effectively neutral once the stronger schedule was already in place.
- `m04_large_shape_acf.py`: selective ACF avoided the small-shape regression from round 1, but compiler guidance still could not compensate for the seed's serialized chunk mapping.

## Next-round hypothesis
The search space should now center on the explicit chunk-axis structure:
- keep the chunk-explicit layout
- tune its per-shape `num_warps`, `num_stages`, and possibly descriptor or ACF choices on top of that new structure
- consider whether K and V should diverge only after the chunk-axis rewrite is fixed as the new baseline
