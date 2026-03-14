# Gated DeltaNet chunk_fwd_h — Manual Mutation Iteration 02

## Round-1 result summary
Baseline: `42.5667 us`

Valid children:
- `m01_conservative_warp_stage`: `31.3333 us`
- `m02_deeper_pipeline`: `31.5 us`
- `m04_large_shape_descriptor`: `31.3 us` — winner
- `m05_large_shape_acf`: `31.3333 us`

Invalid child:
- `m03_explicit_chunk_layout`: correctness failure on all tests

## What round 1 taught us
- The dominant win is in schedule/config tuning, not a large structural rewrite.
- Large-shape `tensor_descriptor` indexing is slightly better than schedule-only tuning.
- Large-shape ACF is competitive, so compiler guidance may stack with the descriptor winner.
- The explicit chunk-layout child broke recurrence semantics and should be excluded until re-derived carefully.

## Hypotheses for round 2
1. Keep the round-1 descriptor winner and add ACF only on the large benchmark shapes.
2. Keep the descriptor winner and deepen only the 1024 benchmark pipeline.
3. Keep the descriptor winner and deepen both large benchmark shapes by one stage.
4. Keep the descriptor winner but use ACF only on the 1024 benchmark.
5. Keep the descriptor winner but use ACF only on the 512 benchmark.
6. Keep the descriptor winner and raise only the 512-shape pipeline to 3 stages.
7. Keep the descriptor winner and raise only the 1024-shape warp count to 16.
8. Keep the descriptor winner and apply descriptor indexing to the `V=128` test shape as well.
9. Keep the large-shape ACF child but add descriptor only on the 1024 benchmark.
10. Keep the conservative warp/stage child and add descriptor only on the 1024 benchmark.

## Discarded on theory
- **H7** — `num_warps=16` is too aggressive for this exploit round.
- **H8** — test-shape descriptor does not target benchmark runtime directly.
- **H9** — mixes ACF and descriptor asymmetrically without stronger motivation.
- **H10** — overlaps too closely with the descriptor winner family.
- **H6** — 512-only stage increase is weaker than a broader deeper-pipeline exploit.

## Retained
- descriptor + large-shape ACF
- descriptor + 1024-only deeper pipeline
- descriptor + both-large-shape deeper pipeline
- descriptor + 1024-only ACF
- descriptor + 512-only ACF

## Distilled round-2 variants
- `m01_descriptor_plus_acf.py`
- `m02_descriptor_deeper_1024.py`
- `m03_descriptor_deeper_both.py`
- `m04_descriptor_acf_1024_only.py`
- `m05_descriptor_acf_512_only.py`

## Round-2 objective
Beat `31.3 us` while preserving correctness, or at minimum identify whether ACF or deeper pipelining stacks better on top of the descriptor winner.
