# Gated DeltaNet recompute_w_u — Manual Mutation Iteration 03

## New baseline
Promoted baseline: explicit chunk-axis kernel from round 2.

Measured baseline:
- `mean_runtime_us`: `27.866666666666664`
- `min_runtime_us`: `17.1`

Per-shape baseline:
- `(1, 64, 1, 64, 64)`: `17.1 us`
- `(2, 512, 3, 64, 64)`: `32.0 us`
- `(2, 1024, 3, 64, 64)`: `34.5 us`

## Step 2 — 10 hypotheses
1. Increase `num_stages` on the chunk-axis baseline for the 512 and 1024 benchmark shapes.
2. Add `indexing="tensor_descriptor"` to the chunk-axis baseline on benchmark shapes.
3. Add `advanced_controls_file` only on the large benchmark shapes of the chunk-axis baseline.
4. Reduce `block_d` from `64` to `32` on benchmark shapes to reduce register pressure and improve concurrency.
5. Use `block_d=32` only on the 1024 benchmark shape.
6. Combine deeper staging with descriptor indexing on the chunk-axis baseline.
7. Combine deeper staging with ACF on the chunk-axis baseline.
8. Increase only the V-path pipeline depth because the V test surface includes `V=128`.
9. Use descriptor indexing only on the 1024 benchmark shape.
10. Use K-path and V-path config divergence on benchmark shapes even when `K=V=64`.

## Step 3 — Discard 5 hypotheses on theory
Discarded:
- **H8** — benchmark shapes all use `V=64`, so V-path-only tuning is weak for this round.
- **H9** — descriptor only on 1024 is too narrow compared with the broader benchmark-shape descriptor test.
- **H10** — K/V divergence on equal benchmark widths is low-value until a stronger signal appears.
- **H5** — 1024-only block shrink is subsumed by the broader `block_d=32` benchmark hypothesis.
- **H7** — staging plus ACF compounds two levers before either is proven on the new baseline.

Retained:
- **H1** deeper pipeline
- **H2** descriptor indexing
- **H3** large-shape ACF
- **H4** smaller `block_d`
- **H6** deeper pipeline + descriptor

## Step 4 — Set-level gaps
The retained set still misses:
- one pure exploit child very close to the new baseline
- one higher-risk combination child to test whether descriptor and deeper staging stack

## Step 5 — Two extra hypotheses
11. Pure exploit: deepen only the 512-shape pipeline and keep everything else at the promoted baseline.
12. Combination: use descriptor indexing plus deeper staging on the 512 and 1024 shapes.

## Distilled round-3 ideas
- `m01_pipeline_512_focus.py`
- `m02_pipeline_both_large.py`
- `m03_descriptor_large.py`
- `m04_block32_large.py`
- `m05_descriptor_pipeline_combo.py`

## Goal
Beat the promoted `27.87 us` baseline, or prove that the current win is already near the local optimum for the present Helion surface.

## Round-3 outcome
- Promoted baseline reproduced at `27.766666666666666 us`
- `m04_block32_large.py` won at `27.366666666666664 us`
- all other round-3 children were flat within noise

Per-shape picture for the round-3 winner:
- `(1, 64, 1, 64, 64)`: `16.9 us`
- `(2, 512, 3, 64, 64)`: `30.5 us`
- `(2, 1024, 3, 64, 64)`: `34.7 us`

## What round 3 says
- The chunk-axis rewrite remains the dominant improvement.
- Smaller `block_d` on the large benchmark shapes produced the only further gain.
- Additional stage depth and descriptor indexing appear saturated on top of the chunk-axis baseline.
