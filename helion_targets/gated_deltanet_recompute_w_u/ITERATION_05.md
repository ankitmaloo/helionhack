# Gated DeltaNet recompute_w_u - Manual Mutation Iteration 05

## Baseline
Promoted baseline after round 4:
- fused dual-output chunk kernel
- shared `A_c` load across `w` and `u`
- current measured baseline: `25.666666666666668 us`

## Step 2 - 20 wild ideas
1. Remove `_chunk_a`, `_chunk_x`, and `_unchunk_x` entirely and run the fused kernel directly on original layout.
2. Keep direct layout, but still pre-scale `k` and `v` on the host.
3. Keep direct layout and move only the `beta` / `beta*exp(g)` multiplies into the kernel.
4. Keep direct layout and move both the multiplies and `exp(g)` into the kernel.
5. Keep chunked layout but write outputs directly to final `[B, T, H, D]` layout.
6. Replace the two inner `hl.dot` calls with one dot over `torch.cat([k, v], dim=-1)`.
7. Do the concatenated single-dot kernel in direct layout.
8. Do the concatenated single-dot kernel in chunked layout.
9. Remove `.contiguous()` from the chunk transforms and test whether view-based chunking is enough.
10. Add `tensor_descriptor` to the direct-layout fused kernel.
11. Add ACF to the direct-layout fused kernel.
12. Add `persistent_interleaved` to the direct-layout fused kernel.
13. Revisit block-size search on the direct-layout fused kernel.
14. Revisit warp-count search on the direct-layout fused kernel.
15. Process two chunks per program in the direct-layout kernel.
16. Process multiple heads per program in the direct-layout kernel.
17. Split the reduction dimension into two 32-row subtile loads and accumulate both outputs interleaved.
18. Flatten `(chunk, head)` only after removing host-side chunk materialization.
19. Specialize the large benchmark shapes to a different structural body than the tests.
20. Try `block_ptr` on the direct-layout fused kernel.

## Step 3 - Prune to 10
Discarded on theory:
- **H5** direct-final-write on chunked inputs is weaker than removing chunked inputs entirely.
- **H14** warp-only retuning is lower value than changing dataflow.
- **H15** two chunks per program is harder to express cleanly without risking attribution issues.
- **H16** multi-head ownership is less direct than direct-layout cleanup.
- **H17** reduction-subtile accumulation is attractive but more invasive than the simpler direct-layout rewrites.
- **H18** mapping-only linearization already looked weak before; it should wait until after stronger dataflow changes.
- **H19** shape-split structural bodies add complexity before the simpler direct-layout variants are measured.
- **H20** `block_ptr` is lower priority than descriptor on this regular layout.
- **H13** block-only retuning is retained only when attached to the direct-layout body.
- **H2** host pre-scaling in direct layout is kept as one control variant, not the core mutation axis.

Retained from theory:
- **H1** direct-layout fused kernel
- **H3** direct-layout with in-kernel scale multiplies
- **H4** direct-layout with full in-kernel `exp`
- **H6** concatenated single-dot kernel
- **H7** direct-layout concatenated single-dot kernel
- **H8** chunked concatenated single-dot kernel
- **H9** view-based chunking without `.contiguous()`
- **H10** direct-layout + descriptor
- **H11** direct-layout + ACF
- **H12** direct-layout + persistent scheduling

## Step 4 - Set-level gaps from subagent
The subagent identified these remaining structural gaps:
- host-side chunking and unchunking still materialize transformed tensors
- host-side scaled temporaries for `k` and `v` still exist
- the fused kernel still executes two separate output-dimension loops
- outputs are still written in chunked layout and restored later
- mapping changes matter less than cutting data movement, but direct-layout plus mapping was never tested

## Step 5 - Five extra hypotheses from the gaps
21. Use direct layout with host pre-scaling as a control to isolate just the chunk/unchunk removal.
22. Use direct layout plus descriptor to see whether TMA matters more once chunk transforms are gone.
23. Use direct layout plus ACF to see whether compiler guidance matters more on the new structure.
24. Use direct layout plus persistent scheduling only, without other structural changes.
25. Use direct layout plus in-kernel scaling as the minimal "no 4D temporaries" rewrite.

## Distilled 10 mutations to run
- `m01_direct_prescaled.py`
- `m02_direct_prescaled_block64_large.py`
- `m03_direct_prescaled_descriptor.py`
- `m04_direct_prescaled_acf.py`
- `m05_direct_load_scale.py`
- `m06_direct_load_scale_exp_inside.py`
- `m07_concat_chunked_single_dot.py`
- `m08_concat_direct_single_dot.py`
- `m09_view_chunked_nocontig.py`
- `m10_direct_prescaled_persistent.py`

## Goal
Push below the `25.67 us` fused baseline by removing remaining host-side layout work and reducing duplicated inner-loop work inside the kernel body.

## Results
Run summary:
- run dir: `/Users/ankit/Documents/dev/hackathon/pytorchkernel/helion_targets/manual_runs/gated_deltanet_recompute_w_u_round5_codex`
- best child: `m06_direct_load_scale_exp_inside.py`
- best mean runtime: `7.933333333333334 us`

Key outcomes:
- `m06_direct_load_scale_exp_inside.py` was the clear winner. It kept the direct-layout kernel, removed all chunk/unchunk transforms, removed scaled temporaries, and computed `beta * exp(g)` inside the kernel. Benchmarks dropped to `6.9 us`, `8.9 us`, and `8.0 us`.
- `m05_direct_load_scale.py` was the second major win at `11.8 us`. This shows that removing the 4D scaled temporaries alone is already a huge gain, even before moving `exp(g)` inside the kernel.
- `m01_direct_prescaled.py` immediately cut the old fused baseline from `25.73 us` to `16.47 us`, proving that direct layout and final-layout writes were the first missing structural lever.
- `m02_direct_prescaled_block64_large.py` improved that direct-layout control to `16.2 us`, showing that the large-shape block-size optimum moved back to `64` once the old chunked-temp pipeline was removed.
- `m04_direct_prescaled_acf.py` helped a bit on top of the direct-layout control, reaching `16.07 us`, mainly by improving the 1024 benchmark.
- `m09_view_chunked_nocontig.py` landed at `17.03 us`, which is strong evidence that `.contiguous()` copies were a real cost, but still not enough by itself to beat the direct raw-input kernels.
- `m07_concat_chunked_single_dot.py` regressed badly to `28.5 us`, and `m08_concat_direct_single_dot.py` was only `19.17 us`, so reducing the number of `hl.dot` calls was not the key bottleneck.
- `m10_direct_prescaled_persistent.py` was unstable and slower at `20.43 us`, so persistent scheduling is not the next frontier here.

Conclusion:
- The path to sub-10 us was structural, not scheduler-level.
- The biggest wins came in order: remove chunk/unchunk materialization, remove scaled temporaries, then move the gate computation inside the direct-layout kernel.
- The promoted baseline after round 5 should be `m06_direct_load_scale_exp_inside.py`.
