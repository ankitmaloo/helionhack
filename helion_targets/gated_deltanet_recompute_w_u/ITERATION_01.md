# Gated DeltaNet recompute_w_u — Manual Mutation Iteration 01

## Baseline
Target: `helion_targets/gated_deltanet_recompute_w_u/seed.py`

Primary goal:
- lower `mean_runtime_us`

Primary layer:
- chunk-matmul execution layer

Measured baseline from the round-1 control run:
- `mean_runtime_us`: `49.666666666666664`
- `min_runtime_us`: `17.0`

## Step 2 — 10 hypotheses
1. Increase benchmark-shape `num_warps` and `num_stages` while leaving kernel math unchanged.
2. Switch benchmark shapes to `indexing="tensor_descriptor"` for more efficient descriptor-based loads on B200.
3. Use `flatten_loops=True` plus `pid_type="persistent_interleaved"` to better schedule many small independent chunk matmuls.
4. Make chunk parallelism explicit by reshaping inputs to `[B, NT, H, C, D]` and tiling over `NT` directly.
5. Hardcode a `recompute_w_u_fwd_*.acf` file for benchmark shapes to test compiler-guided scheduling.
6. Fuse `w` and `u` into one kernel so `A` is loaded once per chunk.
7. Move host-side prescaling into the kernel body so elementwise scaling can be fused with the matmul load path.
8. Use distinct K-path and V-path configs instead of the same config tuple for both kernels.
9. Add more aggressive register-pressure controls through smaller `block_d` and deeper staging on the V-path.
10. Reorder the outer loop structure without explicit chunk reshaping, using only flattening and config changes.

## Step 3 — Discard 5 hypotheses on theory
Discarded:
- **H6** — fused dual-output computation is too invasive for round 1 and makes attribution harder.
- **H7** — moving prescaling into the kernel changes both the dataflow and the schedule at once.
- **H9** — register-pressure tuning is relevant, but it is weaker than testing chunk exposure and compiler guidance first.
- **H10** — loop reordering without explicit chunk exposure overlaps too much with the stronger `flatten_loops` and persistent-scheduling hypothesis.
- **H8** — K/V-specific config split is useful, but it is better treated as an implementation detail inside stronger config hypotheses than as a standalone idea.

Retained from theory:
- **H1** conservative warp/stage tuning
- **H2** tensor descriptor indexing
- **H3** persistent scheduling with flattened outer loops
- **H4** explicit chunk-parallel structure
- **H5** ACF-guided compilation

## Step 4 — Set-level gaps
The retained set still misses:
- a conservative control mutation that changes almost nothing except benchmark config pressure
- a compiler-guided child to test whether PTXAS guidance matters more than schedule changes

## Step 5 — Two extra hypotheses
11. Conservative control: keep the kernel body unchanged and raise only benchmark-shape `num_warps` / `num_stages`.
12. Compiler-augmented descriptor path: pair `tensor_descriptor` with an ACF on benchmark shapes.

## Distilled first-round ideas
### Main idea A
Conservative benchmark config tuning.

### Main idea B
Tensor descriptor indexing.

### Main idea C
Flattened persistent scheduling.

### Main idea D
Explicit chunk-parallel structure.

### Exploratory extra
ACF-guided descriptor tuning.

## Manual mutations to implement
Five tracked files:
- `m01_conservative_warp_stage.py`
- `m02_tensor_descriptor.py`
- `m03_persistent_flatten.py`
- `m04_explicit_chunk_axis.py`
- `m05_acf_descriptor.py`

## Mapping from hypotheses to files
- `m01_conservative_warp_stage.py` tests H1 + H11.
- `m02_tensor_descriptor.py` tests H2.
- `m03_persistent_flatten.py` tests H3.
- `m04_explicit_chunk_axis.py` tests H4.
- `m05_acf_descriptor.py` tests H5 + H12.

## Evaluation plan
- run the baseline seed as the control
- run all five mutations in parallel over SSH
- isolate each child in its own copied remote problem directory
- rank by validity first, then `mean_runtime_us`

## Round-1 success criteria
A child is informative if it:
- passes correctness and improves the baseline
- or fails in a way that clearly narrows the next-round search space

## Round-1 outcome
- `m01_conservative_warp_stage.py` won with `42.86666666666667 us`
- `m02_tensor_descriptor.py` was effectively tied at `42.900000000000006 us`
- `m05_acf_descriptor.py` improved only modestly to `47.96666666666667 us` and regressed the small-shape benchmark
- both `flatten_loops` candidates failed before correctness because the remote Helion config spec expected zero `flatten_loops` entries for this kernel
