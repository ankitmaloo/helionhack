# FP8 Per-Token-Group Quantization — Helion Kernel

## What this kernel does
For each group of `group_size` contiguous elements in the input tensor:
1. `absmax = max(|x_group|)`
2. `scale = max(absmax, 1e-10) / 448.0`
3. `x_q = clamp(x / scale, -448.0, 448.0)`

Input: `(x, x_q, x_s)` where x is [num_tokens, hidden_dim], x_q is pre-allocated output, x_s is per-group scales.
Output: `(x_q, x_s)` — quantized values and scale factors.

The kernel reshapes to [N, group_size] where N = num_tokens * num_groups, so each row is one group.

## Hardware: NVIDIA B200
- 183 GB HBM3e
- Compute capability sm_100
- ACF files available at /opt/booster_pack/fp8_group_quant_*.acf (7 files)

## Helion DSL Quick Reference
- `@helion.kernel(config=helion.Config(...), static_shapes=True)` — kernel decorator
- `hl.tile(dim)` — tile iterator, preserves rank
- `hl.specialize(dim)` — compile-time constant dimension
- `hl.register(shape, dtype)` — register-resident tensor
- `torch.amax(x, dim)` — reduction (compiles to Triton reduce)
- `torch.clamp(x, min, max)` — elementwise clamp

## Full helion.Config Space

### Core knobs
- `block_sizes`: list of tile sizes for each `hl.tile()` call. Powers of 2. For this kernel there is one `hl.tile(N)` so it's a single-element list e.g. `[64]`.
- `num_warps`: 1, 2, 4, 8, or 16. Controls thread parallelism per block.
- `num_stages`: pipeline stages (1-5). Software pipelining depth for memory loads.

### Indexing mode
- `indexing`: `"pointer"` (default), `"block_ptr"`, or `"tensor_descriptor"`
  - `"pointer"` — standard pointer arithmetic, most compatible
  - `"block_ptr"` — uses Triton block pointers, can enable better coalescing
  - `"tensor_descriptor"` — TMA descriptor-based loads (B200 sm_100 supports this natively), potentially fastest for aligned accesses

### Loop & tiling controls
- `flatten_loops`: bool — fuse multi-dimensional tile loops into a single 1D loop. Can improve occupancy for simple kernels.
- `reduction_loops`: list of ints — roll reductions into explicit loops instead of full unrolling. E.g. `[512]` means reduce in chunks of 512. Useful when group_size is large and register pressure is high.
- `loop_orders`: list of list of ints — reorder iteration dimensions. E.g. `[[1, 0]]` swaps tile iteration order.
- `range_unroll_factors`: list — unroll factors for each loop range
- `range_num_stages`: list — per-loop software pipelining stages
- `range_multi_buffers`: list — per-loop multi-buffering
- `range_flattens`: list — per-loop flattening
- `range_warp_specializes`: list — per-loop warp specialization

### Parallelism controls
- `pid_type`: `"flat"` (default), `"xyz"`, `"persistent_interleaved"`, `"persistent_blocked"`
  - `"flat"` — 1D grid, one tile per program ID
  - `"xyz"` — multi-dim grid mapping
  - `"persistent_interleaved"` — persistent kernel, threads loop over tiles in interleaved order. Good for small workloads with high launch overhead.
  - `"persistent_blocked"` — persistent kernel, tiles in blocked order. Better L2 locality.
- `l2_grouping`: int — group program IDs for L2 cache locality. E.g. `l2_grouping=8` means 8 adjacent PIDs access nearby memory.

### CompileIQ / ACF (Advanced Controls Files)
- `advanced_controls_file`: path to ACF file — pre-tuned PTXAS compiler configurations
- Available for this kernel: `/opt/booster_pack/fp8_group_quant_*.acf` (7 files, numbered 0-6)
- ACF can provide 5-15% speedup on top of algorithmic optimization
- Example: `helion.Config(advanced_controls_file="fp8_group_quant_6.acf", block_sizes=[4], num_stages=2, num_warps=8, reduction_loops=[512])`
- Try all 7 ACF files — performance varies by config

### TileIR backend (experimental)
- Enable with env vars: `ENABLE_TILE=1 HELION_BACKEND=tileir`
- Adds extra config fields: `num_ctas`, `occupancy`
- Caveats: `inline_asm_op` may fail, elementwise ops may be slower

## Optimization Strategy for FP8 Quant

### This is a memory-bound kernel
- One read (x), two writes (x_q, x_s) per element, plus reduction (amax)
- Memory bandwidth is the bottleneck, not compute
- Goal: maximize memory throughput via coalescing, occupancy, and pipelining

### Config tuning priorities (ordered by expected impact)
1. **block_sizes**: Larger = more rows per thread block = better occupancy. Try [32], [64], [128], [256]. For small N, match block_size to N.
2. **num_warps**: For memory-bound, try 4 or 8. More warps = more memory requests in flight.
3. **num_stages**: Try 2-4 for pipelining memory loads. Higher helps when block_sizes is large.
4. **indexing**: Try `"tensor_descriptor"` on B200 — TMA can significantly speed up aligned loads.
5. **pid_type**: Try `"persistent_interleaved"` to reduce launch overhead for smaller shapes.
6. **ACF files**: Apply each of the 7 ACF files to the best config — pick the fastest.
7. **reduction_loops**: If group_size=128, reduction fits in registers. If register pressure is high, try `[64]`.

### Algorithmic opportunities
- The reduction (amax) is over group_size (64 or 128) — fits entirely in registers
- `hl.specialize(x.size(1))` already makes group_size a compile-time constant
- Vectorized loads: group_size is always power-of-2 and aligned — enables vector loads
- Fused scale computation: scale = absmax / 448.0 is trivially fused with the quantization

### Per-shape strategy
- Small shapes (1×256, 4×512): Few groups → small N → small block_sizes, fewer warps
- Medium shapes (16×1024, 8×4096): Moderate N → moderate block_sizes
- Large shapes (256×4096, 256×8192, 4096×7168): Many groups → large block_sizes, more warps, pipelining matters

## Debug & Iteration Workflow
- `HELION_INTERPRET=1` — run kernel in Python interpreter (slow but debuggable)
- `HELION_AUTOTUNE_EFFORT=none` — skip autotuning, use provided config as-is
- `HELION_AUTOTUNE_EFFORT=quick` — light autotuning sweep
- `HELION_AUTOTUNE_EFFORT=full` — exhaustive autotuning (slow)

## Constraints
- Must use Helion DSL (inline Triton/ASM ≤30% LOC)
- Must pass all test shapes for correctness (rtol=1e-3, atol=1e-3)
- Config must be hardcoded per shape in SHAPE_CONFIGS dict
