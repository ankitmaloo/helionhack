# FP8 Quant Skill — Schedule and Quantization Layer

## Objective
Improve the `fp8_quant` Helion kernel by reducing benchmark runtime on B200 while preserving correctness for all test shapes.

## Selected code layer
The primary layer for the first mutation round is the kernel schedule/config layer.

This includes:
- `block_sizes`
- `num_warps`
- `num_stages`
- `indexing`
- `pid_type`
- `num_sm_multiplier`
- `reduction_loops`
- `advanced_controls_file`

The secondary layer is the quantization math inside the kernel body, but only through small algebraically equivalent rewrites.

## Why this layer first
- The seed kernel is already correct.
- The kernel is mostly memory-bound.
- The biggest low-risk wins are likely to come from launch geometry, staging, indexing mode, and compiler hints.
- Large algorithmic rewrites are harder to attribute and more likely to break correctness.

## Baseline behavior
The seed implementation does:
1. reshape input into `[N, group_size]`
2. compute per-row `absmax`
3. compute `scale = absmax / 448.0`
4. quantize with clamp
5. write quantized values and scales back

Known baseline metrics from prior evaluation:
- `mean_runtime_us`: `32.16666666666667`
- `min_runtime_us`: `7.3`
- `correctness`: `1.0`

## Invariants
These must remain true:
- `x_q` and `x_s` shapes and semantics must not change.
- `x_s` must store the true per-group scale, not an inverse scale.
- Quantized values must still represent `clamp(x / scale, -448, 448)`.
- The kernel must remain implemented in Helion DSL.
- The shape dispatch table must still cover all required test and benchmark shapes.
- Output tolerance must continue to satisfy the existing evaluator thresholds.

## Safe mutation surface
Low-risk edits:
- tune `helion.Config` per shape
- use `indexing="tensor_descriptor"` or `indexing="pointer"`
- change `pid_type`
- change `num_warps` and `num_stages`
- change `block_sizes`
- add `advanced_controls_file`
- add `reduction_loops` for large group sizes
- replace division by scale with multiplication by reciprocal if `x_s` still stores the true scale

## High-risk edits to avoid in round 1
- changing the external API of `custom_kernel`
- removing per-shape specialization
- introducing inline Triton or ASM
- changing output dtype semantics
- changing the mathematical definition of `scale`
- introducing multiple unrelated algorithmic rewrites in one candidate

## Likely performance levers
### Large-shape throughput
The benchmark shapes are large enough that occupancy, memory pipelining, and indexing strategy should dominate.

### Indexing mode
B200 may benefit from `tensor_descriptor` for aligned accesses.

### Persistent scheduling
Persistent scheduling may help when launch overhead or inter-tile scheduling becomes material.

### ACF use
A B200-specific advanced controls file may unlock extra performance once a stable config is found.

### Register pressure
For `group_size=128`, `reduction_loops` may help if the chosen config is register-limited.

## Measurement protocol
For every mutation, record:
- `valid`
- `passed_correctness`
- `mean_runtime_us`
- `min_runtime_us`
- `compile_time_s`
- `test_time_s`
- `benchmark_time_s`
- `failure_reasons`

Compare every valid mutation to the seed baseline first, then to the best mutation in the current round.

## Mutation design rules
- Prefer one dominant idea per mutation.
- Keep at least one conservative control mutation.
- Use separate files for separate hypotheses.
- If running in parallel, isolate the remote problem directory per variant.
- Treat compile or correctness failures as information about the hypothesis, not just implementation errors.
