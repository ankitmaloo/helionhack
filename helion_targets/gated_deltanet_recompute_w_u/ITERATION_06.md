# Gated DeltaNet recompute_w_u - Manual Mutation Iteration 06

## Purpose
Round 6 starts with a next-state prediction model instead of a broad mutation sweep.

The kernel is already in the sub-10 us band, so the next useful unit of work is:
- predict what is still taking time
- define how to validate or invalidate that prediction
- define a bounded attack surface before writing more mutations

## Current baseline
Promoted baseline after round 5:
- direct-layout fused kernel
- no chunk/unchunk materialization
- no pre-scaled 4D temporaries
- `beta * exp(g)` computed inside the kernel

Current baseline file:
- `seed.py`

Current measured runtime:
- mean: `7.933333333333334 us`
- benchmarks:
  - `(1, 64, 1, 64, 64)`: `6.9 us`
  - `(2, 512, 3, 64, 64)`: `8.9 us`
  - `(2, 1024, 3, 64, 64)`: `8.0 us`

## State transition from round 5
Observed runtime transitions:
- old fused chunked baseline: `25.73 us`
- direct layout, still host pre-scaled: `16.47 us`
- direct layout, in-kernel scale but precomputed gate: `11.8 us`
- direct layout, in-kernel `exp(g)`: `7.93 us`

Interpretation:
- removing chunk/unchunk materialization removed one full cost class
- removing scaled temporaries removed a second cost class
- moving `exp(g)` inside the kernel removed or hid another cost class
- persistent scheduling and concat-dot rewrites did not help

## Step-by-step execution model
Current kernel body in `seed.py` executes these steps per `(B, 64-token tile, H)`:

1. Program setup and tile ownership
   - outer tile over `[batch, seqlen, nheads]`
   - predicted cost: low, but no longer negligible at sub-10 us

2. Load `A` tile and cast to fp32
   - `b_A = A[i_b, tile_t, i_h, :].to(acc_dtype)`
   - predicted cost: high
   - reason: shared across both outputs and always paid once per tile

3. Load `beta` and `g`, cast, compute gate factors
   - `beta_t = beta[...]`
   - `gate_t = beta_t * exp(g[...])`
   - predicted cost: medium-high
   - reason: this is now explicitly in the hot loop, and the round-5 delta from `m05` to `m06` says it still mattered

4. `w` path inner loop
   - load `k`
   - cast to fp32
   - multiply by `gate_t`
   - run `hl.dot(b_A, ...)`
   - cast/store
   - predicted cost: very high

5. `u` path inner loop
   - load `v`
   - cast to fp32
   - multiply by `beta_t`
   - run `hl.dot(b_A, ...)`
   - cast/store
   - predicted cost: very high

6. Output writes
   - write `out_w`
   - write `out_u`
   - predicted cost: medium
   - reason: final-layout writes are much cheaper than before, but two output tensors still double store traffic

## Ordered cost prediction
Predicted remaining cost buckets, from highest to lowest:

1. Two separate dot pipelines plus their surrounding load/cast/scale/store traffic
2. `A` tile load and cast
3. Gate formation: `beta`, `g`, `exp(g)`, broadcast
4. Output store traffic across `w` and `u`
5. Program setup and scheduling overhead

## Falsifiable hypotheses
### H1
The two separate inner loops are now the main structural tax.

Validation:
- a rewrite that shares more of the reduction path across `w` and `u` beats the current kernel by a clear margin on large shapes

Invalidation:
- such a rewrite is flat or regresses, like the concat-dot family

### H2
The current promoted body is still using stale large-shape configs.

Validation:
- the current `exp-inside` kernel improves with `block64` or another direct-layout-specific retune on `(512, 1024)` benchmarks

Invalidation:
- shape-local config changes on the current kernel are flat within noise

### H3
Per-tile gate math is still a first-order hot-path cost.

Validation:
- a gate-focused rewrite that changes how `beta`, `g`, or `exp(g)` is formed beats the current kernel without changing the overall dataflow

Invalidation:
- gate-only rewrites are flat, implying that gate math is already hidden behind the main math pipeline

### H4
Cast-and-scale traffic around `k` and `v` is still more important than reducing the raw number of `hl.dot` calls.

Validation:
- rewrites that reduce `.to(acc_dtype)` or fuse scaling deeper into the load path help

Invalidation:
- dot-structure rewrites help more than load/cast rewrites

### H5
Output-store pressure is now measurable.

Validation:
- a rewrite that coarsens output ownership or reduces intermediate stores improves the large benchmark shapes

Invalidation:
- store-focused rewrites are flat, meaning reads and math still dominate

### H6
Program ownership is still too fine-grained, but only after keeping the current winning direct-layout dataflow intact.

Validation:
- a narrow ownership change such as chunk-pair or multi-head-per-program beats the current kernel

Invalidation:
- these rewrites are again flat or unstable, confirming that scheduling is not the frontier

### H7
The kernel is more 512-shape limited than 1024-shape limited.

Validation:
- shape-local changes improve the `512` benchmark more than the `1024` benchmark

Invalidation:
- gains come mostly from `1024` or both shapes move together

## Attack surface
### Attack surface 1: inner loop body
File region:
- `seed.py`, current body from the outer tile loop through the `tile_k` and `tile_v` loops

Why:
- this is where the two dominant dot pipelines still live

Mutations that fit:
- shared-reduction rewrite
- interleaved `w/u` accumulation
- deeper reuse of loaded `A` subtile
- reduction-subtile accumulation

### Attack surface 2: gate construction
File region:
- `seed.py`, `beta_t` and `gate_t`

Why:
- this is the only hot-path scalar math that still clearly moved performance in round 5

Mutations that fit:
- gate formation reordering
- gate hoisting or split-path gate handling
- alternate staging of `beta` and `exp(g)`

### Attack surface 3: large-shape config table
File region:
- `seed.py`, `SHAPE_CONFIGS`

Why:
- round 5 already showed that direct layout shifted the best tile shape
- the promoted baseline still uses inherited `[32, 32]` large-shape configs

Mutations that fit:
- `block64` on large shapes
- ACF on top of the promoted body
- stage and warp repair only on the promoted body

### Attack surface 4: program ownership
File region:
- outer `hl.tile([batch, seqlen, nheads], ...)`

Why:
- lower confidence than the first three, but still the main remaining mapping lever

Mutations that fit:
- chunk-pair ownership
- multi-head ownership
- narrow ownership coarsening that preserves current dataflow

### Attack surface 5: output materialization
File region:
- `out_w` and `out_u` allocation and writes

Why:
- likely lower priority, but now plausible because the kernel is already very fast

Mutations that fit:
- output coarsening
- delayed store strategies
- combined output staging if Helion supports it cleanly

## Negative knowledge to carry forward
- concat-dot is not the main lever
- broad persistent scheduling is not the main lever
- descriptor mode is not the main lever by itself once layout is fixed
- the winning path so far has been dataflow simplification, not generic schedule tuning

## Proposed round-6 shape of work
Before running another 10-child sweep:
- choose 3-4 mutations from attack surfaces 1-3
- choose 1-2 narrow controls from attack surface 4
- make every child falsify one explicit hypothesis

## Immediate prediction
Most likely next win:
- the current kernel body plus a direct-layout-specific large-shape retune
or
- a loop-body rewrite that shares more of the `A`-driven reduction path across `w` and `u`

Less likely next win:
- persistent scheduling
- another concat-dot formulation
- broad descriptor-only changes

## Round-6 batch
Children tied to the current hypotheses:
- `m01_block64_large.py`
  - tests `H2`
  - change: retune promoted body to `block64` on both large benchmark shapes
- `m02_block64_large_acf.py`
  - tests `H2`
  - change: add ACF on top of the `block64` large-shape retune
- `m03_block64_1024_only.py`
  - tests `H7`
  - change: retune only the 1024 benchmark shape to `block64`
- `m04_descriptor_large.py`
  - tests `H2`
  - change: descriptor mode on the promoted body for large shapes
- `m05_same_tile_dual_path.py`
  - tests `H1`
  - change: for `K == V` shapes, compute both paths inside a shared `tile_d` loop instead of two separate outer loops
- `m06_gate_exp_reuse.py`
  - tests `H3`
  - change: factor `exp(g)` explicitly into a local vector before building `gate_t`

## Results
Run summary:
- run dir: `/Users/ankit/Documents/dev/hackathon/pytorchkernel/helion_targets/manual_runs/gated_deltanet_recompute_w_u_round6_codex`
- best child: `m01_block64_large.py`
- best mean runtime: `7.1000000000000005 us`

Outcome by hypothesis:
- `H2` survived.
  - `m01_block64_large.py` won outright.
  - the current promoted body was still using stale large-shape block sizes.
  - the biggest gain came from the `512` benchmark: `8.9 us -> 6.9 us`.
- `H7` survived in a narrower form.
  - `m03_block64_1024_only.py` only improved the `1024` benchmark and stayed near baseline overall.
  - this confirms that the real shape-local mismatch was mostly the `512` case.
- `H2 + ACF` did not beat plain config repair.
  - `m02_block64_large_acf.py` improved versus baseline but lost to `m01`.
- descriptor on the promoted body is still flat.
  - `m04_descriptor_large.py` matched the baseline.
- `H3` did not survive this formulation.
  - `m06_gate_exp_reuse.py` was flat.
- `H1` remains unresolved, not disproven.
  - `m05_same_tile_dual_path.py` was invalid on the remote Helion config spec because that kernel shape only accepted one `block_sizes` entry.

New baseline after round 6:
- current seed should keep the same direct-layout, in-kernel-gating body
- large benchmark shapes should use `block_sizes=[64, 64]`

Interpretation:
- the next-state prediction was directionally right.
- there was still one easy config repair left on the winning body.
- after applying it, the remaining frontier is again inside the loop body rather than in descriptor, ACF, or gate-expression micro-structure.
