# Gated DeltaNet Output Computation (chunk_fwd_o) — Helion Kernel

## What this kernel does
Computes the final output by combining inter-chunk (state-based) and intra-chunk
(attention-based) contributions for Gated DeltaNet (arXiv:2412.06464, ICLR 2025).

The sequence is divided into chunks of BT=64 timesteps. Chunks are independent
(h is pre-computed) so all can be processed in parallel.

For each chunk independently:
  inter = q_c @ h * exp(g)                                     # [C, V]
  g_diff = g[:, None] - g[None, :]                             # [C, C]
  qk = causal_mask(q_c @ k_c^T * exp(g_diff))                 # [C, C]
  intra = qk @ v_new_c                                         # [C, V]
  output = (inter + intra) * scale                              # scale = K^(-0.5)

Input: `(q, k, v_new, h, g)` where:
- q: [B, T, H, K] float32 — queries
- k: [B, T, H, K] float32 — keys
- v_new: [B, T, H, V] float32 — corrected values
- h: [B, NT, H, K, V] float32 — per-chunk hidden states
- g: [B, T, H] float32 — cumulative gate

Output: [B, T, H, V] float32

Constraint: T must be a multiple of 64. NT = T // 64. scale = K^(-0.5).

## Hardware: NVIDIA B200
- 183 GB HBM3e
- Compute capability sm_100
- ACF files: /opt/booster_pack/chunk_fwd_o_*.acf (7 files, numbered 0-6)

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
- Available: `/opt/booster_pack/chunk_fwd_o_0.acf` through `chunk_fwd_o_6.acf` (7 files)
- 5-15% speedup. Try all 7 — performance varies significantly by config.

## Test & Benchmark Shapes
Tests: (B=1,T=64,H=2,K=64,V=64), (B=2,T=128,H=4,K=64,V=64), (B=1,T=256,H=4,K=64,V=128)
Benchmarks: (B=1,T=64,H=1,K=64,V=64), (B=2,T=512,H=3,K=64,V=64), (B=2,T=1024,H=3,K=64,V=64)

## Optimization Strategy

### Mixed compute + memory pattern
- Inter-chunk: one matmul q@h [C,K]@[K,V] + exp gating — compute
- Intra-chunk: q@k^T [C,K]@[K,C] attention with causal mask, then attn@v [C,C]@[C,V] — compute
- Total: 3 matmuls per chunk + elementwise gating
- Chunks are fully parallel — no sequential dependency

### Config tuning priorities
1. **block_sizes[V]**: V=64 or 128. Tile V dimension for register management.
2. **num_warps**: 4-8 for matmul-heavy kernels.
3. **flatten_loops**: Fuse B, NT, H dimensions for better parallelism.
4. **indexing**: Try `"tensor_descriptor"` for B200.
5. **ACF files**: 7 files available — sweep all of them.
6. **pid_type**: `"persistent_interleaved"` for better utilization with many small chunks.

### Algorithmic opportunities
- K=64 is specialized → all matmul dimensions known at compile time
- C=64 — the [C,C] attention matrix fits in registers
- Causal mask is lower-triangular — can skip upper-triangle computation
- Inter and intra contributions are independent — can be computed in parallel then summed
- exp(g) and exp(g_diff) can be precomputed once per chunk
- See `helion/examples/attention.py` and `flex_attention.py` for attention patterns
- See `helion/examples/gdn_fwd_h.py` for the tiling pattern over [B, H, V]

### Causal masking in Helion
The causal mask (lower triangular) within each chunk can be created with:
- `torch.arange(C)[:, None] >= torch.arange(C)[None, :]`
- Or `torch.tril(torch.ones(C, C))`
The mask zeros out QK scores where query position < key position.

## Debug & Iteration Workflow
- `HELION_INTERPRET=1` — run in Python interpreter
- `HELION_AUTOTUNE_EFFORT=none` — skip autotuning

## Constraints
- Must use Helion DSL (inline Triton/ASM ≤30% LOC)
- Must pass all test shapes (rtol=1e-2, atol=1e-2)
- Config must be hardcoded per shape
