# Causal Depthwise 1D Convolution — Helion Kernel

## What this kernel does
A core component of Mamba/Mamba-2 architectures. Each channel is convolved independently
(depthwise) with causal (left) zero-padding so that output[t] depends only on input[t-W+1:t+1].

For each batch b, channel d, and time t:
  out[b, d, t] = bias[d] + sum_{k=0}^{W-1} weight[d, k] * x[b, d, t - W + 1 + k]
where out-of-bounds values are treated as zero.

Input: `(x, weight, bias)` where:
- x: [B, D, S] float32 input tensor
- weight: [D, W] float32 depthwise conv weights
- bias: [D] float32 bias

Output: [B, D, S] float32

The kernel operates on zero-padded input x_pad: [B, D, S+W-1] (left-padding with W-1 zeros).

## Hardware: NVIDIA B200
- 183 GB HBM3e
- Compute capability sm_100
- ACF files: /opt/booster_pack/causal_conv_*.acf (3 files, numbered 0-2)

## Helion DSL Quick Reference
- `@helion.kernel(config=helion.Config(...), static_shapes=True)` — kernel decorator
- `hl.tile(dim)` — tile iterator, preserves rank
- `hl.specialize(dim)` — compile-time constant dimension
- `hl.register(shape, dtype)` — register-resident tensor
- `hl.zeros([dim1, dim2], dtype)` — zero-initialized register tensor
- `hl.dot(a, b)` — matrix multiply

## Full helion.Config Space

### Core knobs
- `block_sizes`: list of tile sizes for each `hl.tile()` call. Powers of 2.
- `num_warps`: 1, 2, 4, 8, or 16
- `num_stages`: pipeline stages (1-5)

### Indexing mode
- `indexing`: `"pointer"` (default), `"block_ptr"`, or `"tensor_descriptor"`
  - `"tensor_descriptor"` — TMA descriptor-based loads (B200 sm_100), potentially fastest for aligned accesses

### Loop & tiling controls
- `flatten_loops`: bool — fuse multi-dimensional tile loops into a single 1D loop
- `reduction_loops`: list of ints — roll reductions into explicit loops
- `loop_orders`: list of list of ints — reorder iteration dimensions
- `range_unroll_factors`, `range_num_stages`, `range_multi_buffers`, `range_flattens`, `range_warp_specializes`: per-loop tunables

### Parallelism controls
- `pid_type`: `"flat"`, `"xyz"`, `"persistent_interleaved"`, `"persistent_blocked"`
- `l2_grouping`: int — group PIDs for L2 cache locality

### CompileIQ / ACF
- `advanced_controls_file`: path to ACF file
- Available: `/opt/booster_pack/causal_conv_0.acf`, `causal_conv_1.acf`, `causal_conv_2.acf`
- 5-15% speedup on top of algorithmic optimization

## Test & Benchmark Shapes
Tests:  (B=1,D=64,S=64,W=4), (B=2,D=128,S=128,W=4), (B=1,D=256,S=256,W=3), (B=1,D=128,S=64,W=8), (B=4,D=64,S=128,W=4)
Benchmarks: (B=1,D=1536,S=2048,W=4), (B=1,D=2560,S=2048,W=4), (B=1,D=2560,S=4096,W=4)

## Optimization Strategy

### Memory-bound kernel
- Small W (3-8) means very few FLOPs per element — dominated by memory access
- Each output element requires W loads from x_pad + W loads from weight + 1 bias
- Key: maximize memory throughput via coalescing along S dimension

### Config tuning priorities
1. **block_sizes**: Tile D and S dimensions. Larger S tiles = fewer blocks. D tiles should map to warps.
2. **num_warps**: Try 4 or 8 for memory-bound kernels.
3. **indexing**: Try `"tensor_descriptor"` — input is contiguous, aligned access patterns.
4. **pid_type**: Try `"persistent_interleaved"` for smaller shapes.
5. **ACF files**: Apply 3 available ACF files to best config.

### Algorithmic opportunities
- W is small (3-8) and specialized → loop is fully unrolled at compile time
- Weight and bias are shared across S — reuse in registers
- D dimension is depthwise (each channel independent) — perfect for tiling
- Pre-padding in custom_kernel eliminates branch overhead in the kernel
- For large D: consider flattening B×D into one dimension for better occupancy

## Constraints
- Must use Helion DSL (inline Triton/ASM ≤30% LOC)
- Must pass all test shapes (rtol=1e-2, atol=1e-2)
- Config must be hardcoded per shape
