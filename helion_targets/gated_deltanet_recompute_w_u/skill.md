# Gated DeltaNet recompute_w_u Skill — Chunk Matmul Execution Layer

## Objective
Improve the `gated_deltanet_recompute_w_u` Helion submission by reducing benchmark runtime on B200 while preserving correctness for every required test shape.

## Selected code layer
The first mutation round targets the chunk-matmul execution layer.

This includes:
- chunk traversal structure inside the kernel
- per-shape `helion.Config`
- indexing mode
- loop flattening / persistent scheduling
- K-path versus V-path config separation
- ACF-based compiler guidance

This round does not change the problem API or the mathematical definition of `w` and `u`.

## Why this layer first
- The seed is already structurally simple and correct.
- The kernel performs two independent per-chunk matmuls, which suggests schedule and mapping choices matter more than large algebraic rewrites.
- The context already points to B200-specific levers: `flatten_loops`, `tensor_descriptor`, `persistent_interleaved`, and ACF files.
- Chunk independence creates a plausible structural win: make chunk parallelism explicit instead of serializing chunks inside each `(B, H, D)` program.

## Baseline behavior
The seed does:
1. pre-scale `v` on the host to form `v_scaled`
2. pre-scale `k` on the host to form `k_scaled`
3. launch one Helion chunk-matmul kernel for `u = A @ v_scaled`
4. launch one Helion chunk-matmul kernel for `w = A @ k_scaled`
5. inside each kernel, iterate over `t_i` chunks serially for each `(batch, head, dout tile)`

Baseline control file:
- `seed.py`

Known baseline metrics from the first remote control run:
- `mean_runtime_us`: `49.666666666666664`
- `min_runtime_us`: `17.0`
- `correctness`: `1.0`

## Invariants
These must remain true:
- `custom_kernel(data)` must still return `(w, u)` with shapes `[B, T, H, K]` and `[B, T, H, V]`.
- `u` must equal `A @ (v * beta[:, None])` per chunk.
- `w` must equal `A @ (k * (beta * exp(g))[:, None])` per chunk.
- `T` must remain chunked by `CHUNK_SIZE = 64`.
- Output dtype semantics must remain compatible with the evaluator.
- The submission must stay in Helion DSL.
- Shape dispatch must still cover the required test and benchmark shapes.

## Safe mutation surface
Low-risk edits:
- change `num_warps`, `num_stages`, and `block_sizes`
- use `indexing="tensor_descriptor"`
- use `flatten_loops=True`
- use `pid_type="persistent_interleaved"`
- choose different configs for the K and V kernels
- hardcode `advanced_controls_file` values per shape
- reshape tensors in `custom_kernel` if the mathematical mapping is unchanged

## High-risk edits to avoid in round 1
- changing the external signature of `custom_kernel`
- changing chunk size
- changing the mathematical meaning of `A`, `beta`, or `g`
- mixing multiple structural rewrites in one child
- introducing inline Triton or custom CUDA
- relying on unsupported TileIR-only knobs without a separate control path

## Likely performance levers
### Chunk parallelism
The seed serializes chunks inside the kernel body. Exposing the chunk dimension as a tiled axis may improve parallel work distribution across `B * NT * H`.

### Indexing mode
The input tensors are contiguous and regular. `tensor_descriptor` is a credible B200-specific hypothesis.

### Scheduler policy
The workload has many small independent chunk matmuls, which makes `flatten_loops` and `persistent_interleaved` worth testing.

### Compiler guidance
The host provides five `recompute_w_u_fwd_*.acf` files. A compiler-guided child should be present in the first batch.

### K/V asymmetry
`K` is always 64, while `V` can be 64 or 128. Even if benchmark shapes use `V=64`, the test surface still argues for K-path and V-path configs to be independently tunable.

## Measurement protocol
Run the baseline seed and all mutation files through `manual_mutation_runner.py` with isolated remote problem copies.

Record for every child:
- `valid`
- `passed_correctness`
- `mean_runtime_us`
- `min_runtime_us`
- `compile_time_s`
- `test_time_s`
- `benchmark_time_s`
- `failure_reasons`

Compare children to the baseline first, then to the best valid child in the batch.

## Mutation design rules
- One dominant idea per mutation file.
- Keep `seed.py` untouched.
- Use separate source files under `manual_mutations/round1/`.
- Treat compile failures and regressions as evidence for the next iteration, not wasted work.
