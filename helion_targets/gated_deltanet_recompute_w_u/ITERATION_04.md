# Gated DeltaNet recompute_w_u — Manual Mutation Iteration 04

## Baseline
Promoted baseline after round 3:
- explicit chunk-axis kernel
- `block_sizes=[32]` on benchmark shapes `(2, 512, 3, 64, 64)` and `(2, 1024, 3, 64, 64)`

Measured baseline:
- `mean_runtime_us`: `27.566666666666666`
- `min_runtime_us`: `17.5`

## Step 2 — 20 wild ideas
1. Add ACF to both large benchmark shapes on the current winning structure.
2. Add ACF only on the 512 benchmark shape.
3. Add ACF only on the 1024 benchmark shape.
4. Add `tensor_descriptor` on both large benchmark shapes with `block_d=32`.
5. Add `tensor_descriptor` only on the 1024 benchmark shape with `block_d=32`.
6. Switch large benchmark shapes to `block_ptr`.
7. Use `block_d=16` on the 1024 benchmark shape.
8. Use `block_d=16` on both large benchmark shapes.
9. Keep 512 at `block32` and repair 1024 with `block64` plus deeper staging.
10. Keep 512 at `block32` and repair 1024 with `block32 + descriptor`.
11. Give only the `u` path deeper staging or descriptor tuning.
12. Give only the `w` path smaller blocks or higher warp count.
13. Try `num_warps=16` on the 1024 benchmark shape.
14. Try `num_warps=4` on the 512 benchmark shape.
15. Increase `l2_grouping` on the large benchmark shapes.
16. Try `pid_type="persistent_blocked"` on the current chunk-axis structure.
17. Fuse `w` and `u` into one dual-output chunk kernel so `A_c` is loaded once.
18. Move scaling into the kernel body and eliminate host-side `k_c` / `v_c` temporaries.
19. Manually linearize `(nchunks * heads)` into one execution axis.
20. Tile multiple heads or chunks together inside a single program.

## Step 3 — Prune to 10
Discarded on theory:
- **H3** — ACF on 1024 only is narrower than the stronger two-shape and 512-only tests.
- **H6** — `block_ptr` is plausible, but it is weaker than descriptor on this regular contiguous layout.
- **H8** — `block16` on both large shapes is too likely to over-fragment work.
- **H10** — 1024-only descriptor repair overlaps too much with the broader descriptor tests.
- **H14** — lowering 512 to 4 warps is lower value than testing 16 warps on 1024.
- **H15** — `l2_grouping` is interesting but less direct than shape-local config repair.
- **H16** — `persistent_blocked` is harder to attribute than the simpler mapping variants.
- **H18** — moving scaling into the kernel mixes dataflow and scheduling before the simpler fusion test.
- **H20** — multi-head or multi-chunk tiling is less direct than manual linearization.
- **H2** — ACF on 512 only is kept as a later gap-fill candidate instead of a core retained hypothesis.

Retained from theory:
- **H1** ACF on both large shapes
- **H4** descriptor + block32 on both large shapes
- **H5** descriptor + block32 on 1024 only
- **H7** block16 on 1024
- **H9** repair 1024 with block64/stage4
- **H11** tune only the `u` path
- **H12** tune only the `w` path
- **H13** `num_warps=16` on 1024
- **H17** fused dual-output kernel
- **H19** manual `(nchunks * heads)` linearization

## Step 4 — Set-level gaps from subagent
The subagent identified these underexplored gaps:
- ACF on the winning chunk-axis structure has not really been tested.
- Descriptor on top of `block32` has not been tested thoroughly.
- Block-size search is still shallow beyond `64 -> 32`.
- Benchmark-shape K/V divergence remains mostly absent.
- Warp exploration is narrow and `16` warps has not been tested empirically.
- 512 and 1024 are not being repaired independently enough.
- Dataflow-level rewrites like fused `w/u` were postponed, not disproven.
- Program mapping beyond one program per `(B, chunk, H, dtile)` remains underexplored.

## Step 5 — Five extra hypotheses from the gaps
21. Add ACF to `block32` only on the 512 benchmark shape.
22. Repair 1024 with `block32 + descriptor`, while keeping 512 at the current winner.
23. Add only `u`-path descriptor tuning on large benchmark shapes.
24. Add only `w`-path aggressive warp or block tuning on 1024.
25. Use `block16` plus `num_warps=16` on 1024 to explicitly stress the occupancy/register tradeoff.

## Distilled 10 mutations to run
- `m01_acf_block32_both.py`
- `m02_acf_512_only.py`
- `m03_desc_block32_both.py`
- `m04_desc_block32_1024_only.py`
- `m05_asym_51232_102464.py`
- `m06_block16_1024.py`
- `m07_u_path_tuned.py`
- `m08_w_path_warp16.py`
- `m09_fused_dual_output.py`
- `m10_linearized_chunk_head.py`

## Goal
Push below the current `27.57 us` baseline and widen the evidence base enough to know whether the next frontier is config-level, K/V-asymmetric, or a second structural rewrite.

## Results
Run summary:
- run dir: `/Users/ankit/Documents/dev/hackathon/pytorchkernel/helion_targets/manual_runs/gated_deltanet_recompute_w_u_round4_codex`
- best child: `m09_fused_dual_output.py`
- best mean runtime: `25.666666666666668 us`

Key outcomes:
- `m09_fused_dual_output.py` was the only step-change win. It fused the `w` and `u` chunk matmuls into one kernel and reused the same `A_c` load, dropping benchmarks to `14.9 us`, `30.1 us`, and `32.0 us`.
- `m05_asym_51232_102464.py`, `m07_u_path_tuned.py`, and `m08_w_path_warp16.py` each improved slightly into the `27.33-27.40 us` band, but only by schedule-level margins.
- `m10_linearized_chunk_head.py` did not help; manual `(chunk, head)` linearization stayed near `27.5 us`.
- `m02_acf_512_only.py` was the clear failure. It introduced large runtime instability and regressed to `39.63 us`.
- Descriptor and ACF variants on the current winning structure were mostly flat.

Conclusion:
- The next frontier is code structure, not just shape-local parameter tuning.
- Reusing the same `A_c` tile across both outputs matters more than descriptor or warp micro-tuning on the current decomposition.
- The promoted baseline after round 4 should be the fused dual-output kernel.
