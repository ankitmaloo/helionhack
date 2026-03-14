# Gated DeltaNet chunk_fwd_h — Manual Mutation Iteration 03 (Expanded Code Search)

## Pre-expanded baseline
The best result so far remains the round-1 descriptor winner:
- `m04_large_shape_descriptor`: `31.3 us`

Narrow round-2 exploit children did not beat it.

## 20 code-focused hypotheses
1. Remove the boundary mask because all task shapes satisfy `T % 64 == 0`.
2. Load each chunk of `g` once and derive `g_last` from that chunk instead of reloading scalar and vector separately.
3. Replace `torch.where(mask, exp(...), 0)` with direct scale multiplication under the full-chunk invariant.
4. Explicitly chunk inputs into `[B, NT, H, BT, D]` but keep the chunk loop sequential.
5. Explicitly chunk only `g`, leaving `k/w/u` in the original layout.
6. Explicitly chunk `k`, `w`, and `u`, leaving only `g` unchunked.
7. Preload `k/w/u/g` chunk slices into locals before any math each iteration.
8. Separate `v_new_raw` and `v_gated` variables to reduce aliasing and make the dataflow explicit.
9. Delay dtype casts so the state update path stays in `acc_dtype` longer.
10. Compute `exp(g_last)` once and factor gating as `exp(g_last) * exp(-g)`.
11. Use the descriptor winner config but apply the code-path simplification from hypotheses 1-3.
12. Use the descriptor winner config and chunked sequential views from hypothesis 4.
13. Use the schedule-only winner config with the no-mask code rewrite to test whether code beats descriptor.
14. Combine large-shape descriptor with chunk-local loads and explicit `v_new_raw` / `v_gated` separation.
15. Apply ACF only after the no-mask code rewrite, not on the original body.
16. Keep baseline configs but rewrite the chunk loop using explicit chunk index instead of sequence tile indexing.
17. Keep descriptor configs but precompute `g_scale` and `decay` once per chunk.
18. Keep descriptor configs but chunk `g` and `u` only, since those feed the gating/value path directly.
19. Keep descriptor configs but make the 1024-shape path use the rewritten body while smaller shapes keep the original body.
20. Keep descriptor configs but use the rewritten body plus 1024-only ACF.

## First prune to 10
Dropped:
- **H5** — chunking only `g` is too narrow relative to stronger code rewrites.
- **H6** — chunking only `k/w/u` without `g` complicates the loop with weaker motivation.
- **H9** — delayed cast alone is too small and overlaps with stronger local-dataflow hypotheses.
- **H10** — factorized exponentials overlap heavily with the simpler no-mask/local-g rewrite.
- **H13** — schedule-only config is already nearly tied, but the code search should stay anchored to the descriptor winner.
- **H15** — ACF-on-rewritten-body is better treated as a concrete child than a hypothesis family.
- **H16** — explicit chunk index with baseline config is dominated by descriptor-backed variants.
- **H18** — chunking only `g` and `u` is weaker than fully chunked sequential views.
- **H19** — mixed bodies by shape add complexity without first validating the rewritten body globally.
- **H20** — descriptor + rewritten body + 1024-only ACF is a concrete child, not a broad retained family.

Retained after prune:
- **H1** no-mask full-chunk rewrite
- **H2** local `g` chunk load
- **H3** remove `torch.where`
- **H4** fully chunked sequential views
- **H7** local chunk slice loads
- **H8** split `v_new_raw` and `v_gated`
- **H11** descriptor + no-mask rewrite
- **H12** descriptor + chunked sequential views
- **H14** descriptor + local loads + explicit value flow
- **H17** descriptor + precomputed `g_scale` / decay

## 5 set-level gaps
1. No mutation isolates a code rewrite on top of the schedule-only winner.
2. No mutation tests the rewritten body with ACF after correctness-safe simplification.
3. No mutation tests chunked sequential views without descriptor indexing.
4. No mutation isolates the `v_new_raw` / `v_gated` split without also adding chunk-local loads.
5. No mutation tests whether the no-mask rewrite alone is the main win versus chunked layout.

## Final 10 mutations
1. `m01_descriptor_no_mask.py`
2. `m02_descriptor_no_mask_local_g.py`
3. `m03_descriptor_value_flow_split.py`
4. `m04_descriptor_chunked_sequential.py`
5. `m05_descriptor_chunked_sequential_local_loads.py`
6. `m06_schedule_no_mask.py`
7. `m07_schedule_value_flow_split.py`
8. `m08_descriptor_no_mask_acf.py`
9. `m09_descriptor_gscale_precompute.py`
10. `m10_chunked_sequential_baseline.py`

## Why these 10
- they all include a real body/dataflow mutation, not only config tuning
- they span both descriptor-backed and non-descriptor-backed code paths
- they test no-mask simplification, chunk-local loads, explicit value-flow separation, and corrected chunked sequential indexing
- they include a single ACF stack after a correctness-safe rewrite rather than many compiler-only children

## Run objective
Run all 10 mutations in parallel against the baseline seed and identify whether code-path simplification or corrected chunked sequential indexing can break through the current `31.3 us` wall.
