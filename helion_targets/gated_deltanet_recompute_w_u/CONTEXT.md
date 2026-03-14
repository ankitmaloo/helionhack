# Gated DeltaNet WY-Transform Forward (recompute_w_u) — Helion Kernel

## What this kernel does
Computes WY-transformed keys (w) and values (u) for the chunkwise parallel forward
pass of Gated DeltaNet (arXiv:2412.06464, ICLR 2025). One of three per-chunk kernels
in the forward pipeline.

The sequence is divided into non-overlapping chunks of BT=64 timesteps.
For each chunk independently:
  u = A @ (v * beta[:, None])                    # WY-transformed values
  w = A @ (k * (beta * exp(g))[:, None])          # WY-transformed keys

where A is a [BT, BT] WY representation matrix per chunk.

Input: `(k, v, beta, A, g)` where:
- k: [B, T, H, K] float32 — keys
- v: [B, T, H, V] float32 — values
- beta: [B, T, H] float32 — gating coefficients
- A: [B, T, H, BT] float32 — WY matrix (BT=64, last dim is chunk-local index)
- g: [B, T, H] float32 — cumulative gate

Output: `(w, u)` where:
- w: [B, T, H, K] float32 — WY-transformed keys
- u: [B, T, H, V] float32 — WY-transformed values

Constraint: T must be a multiple of 64.

## Hardware: NVIDIA B200
- 183 GB HBM3e
- Compute capability sm_100
- ACF files: /opt/booster_pack/recompute_w_u_fwd_*.acf (5 files, numbered 0-4)

## Full helion.Config Space

### Core knobs
- `block_sizes`: tile sizes for each `hl.tile()` call. Powers of 2.
- `num_warps`: 1, 2, 4, 8, or 16
- `num_stages`: pipeline stages (1-5)

### Indexing mode
- `indexing`: `"pointer"`, `"block_ptr"`, or `"tensor_descriptor"`

### Loop & tiling controls
- `flatten_loops`, `reduction_loops`, `loop_orders`
- Per-loop: `range_unroll_factors`, `range_num_stages`, `range_multi_buffers`, `range_flattens`, `range_warp_specializes`

### Parallelism controls
- `pid_type`: `"flat"`, `"xyz"`, `"persistent_interleaved"`, `"persistent_blocked"`
- `l2_grouping`: group PIDs for L2 cache locality

### CompileIQ / ACF
- `advanced_controls_file`: path to ACF file
- Available: `/opt/booster_pack/recompute_w_u_fwd_0.acf` through `recompute_w_u_fwd_4.acf`
- 5-15% speedup

## Test & Benchmark Shapes
Tests: (B=1,T=64,H=2,K=64,V=64), (B=2,T=128,H=4,K=64,V=64), (B=1,T=256,H=4,K=64,V=128)
Benchmarks: (B=1,T=64,H=1,K=64,V=64), (B=2,T=512,H=3,K=64,V=64), (B=2,T=1024,H=3,K=64,V=64)

## Optimization Strategy

### Compute-bound: two batched matmuls
- u = A @ v_scaled: [BT, BT] @ [BT, V] per chunk — BT=64, V=64 or 128
- w = A @ k_scaled: [BT, BT] @ [BT, K] per chunk — BT=64, K=64
- Chunks are independent — fully parallel across B * NT * H instances
- Total: 2 matmuls per chunk, each 64×64 @ 64×(64 or 128)

### Config tuning priorities
1. **block_sizes**: Tile V and K dimensions for register management.
2. **num_warps**: 4-8 for matmul workloads.
3. **flatten_loops**: Fuse B, NT, H into single dimension for parallelism.
4. **indexing**: `"tensor_descriptor"` for B200 TMA.
5. **ACF files**: 5 files available — sweep all.
6. **pid_type**: `"persistent_interleaved"` for many small independent chunks.

### Algorithmic opportunities
- BT=64 and K=64 are compile-time constants → efficient register matmuls
- A matrix is [BT, BT] = 64×64 — fits in registers
- The scaling (beta, exp(g)) can be fused with the matmul (precompute scaled inputs)
- A is shared between w and u computation — load once, use twice
- Consider fusing both matmuls in a single kernel pass
- For V=128: tile V into 2 blocks of 64 for better register pressure

### A matrix structure
- A is the solved WY representation: lower-triangular structure within each chunk
- A[b, t, h, :] gives the BT weights for how timestep t depends on all timesteps in its chunk
- When tiling with `hl.tile(T, block_size=BT)`, `A[i_b, t_i, i_h, :]` loads [BT, BT]

## Debug & Iteration Workflow
- `HELION_INTERPRET=1` — run in Python interpreter
- `HELION_AUTOTUNE_EFFORT=none` — skip autotuning

## Constraints
- Must use Helion DSL (inline Triton/ASM ≤30% LOC)
- Must pass all test shapes (rtol=1e-2, atol=1e-2)
- Config must be hardcoded per shape
