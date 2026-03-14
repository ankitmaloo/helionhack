# Gated DeltaNet Inter-Chunk State Recurrence (chunk_fwd_h) — Helion Kernel

## What this kernel does
Implements inter-chunk state recurrence for the chunkwise parallel forward pass of
Gated DeltaNet (arXiv:2412.06464, ICLR 2025). This is the sequential bottleneck —
chunks must be processed in order.

The sequence is divided into chunks of BT=64 timesteps. Processing is sequential
across chunks but parallel across (B, H) and within each chunk.

For each (b, h) pair, starting with h_state = zeros(K, V):
  For each chunk c = 0, 1, ..., NT-1:
    1. Store: h_out[b, c, h] = h_state
    2. Compute: v_new = u - w @ h_state     (matmul: [C,K] @ [K,V] = [C,V])
    3. Gate: v_gated[t] = v_new[t] * exp(g[last_t] - g[t])
    4. Decay: h_state = h_state * exp(g[last_t])
    5. Update: h_state = h_state + k^T @ v_gated  (matmul: [K,C] @ [C,V] = [K,V])

Input: `(k, w, u, g)` where:
- k: [B, T, H, K] float32 — keys
- w: [B, T, H, K] float32 — WY-transformed keys
- u: [B, T, H, V] float32 — WY-transformed values
- g: [B, T, H] float32 — cumulative gate

Output: `(h, v_new)` where:
- h: [B, NT, H, K, V] float32 — per-chunk hidden states (BEFORE each chunk)
- v_new: [B, T, H, V] float32 — corrected values

Constraint: T must be a multiple of 64. NT = T // 64.

## Hardware: NVIDIA B200
- 183 GB HBM3e
- Compute capability sm_100
- ACF files: /opt/booster_pack/chunk_fwd_h_*.acf (2 files, numbered 0-1)

## Reference Helion Implementation
See `helion/examples/gdn_fwd_h.py` — a simpler variant that returns only h (without v_new).
Key patterns from that example:
- Tiles over [batch, nheads, dstate] with block_size=[1, 1, block_v]
- Uses `hl.register_block_size(dstate)` for V dimension tiling
- Sequential loop via `hl.tile(seqlen, block_size=chunk_size)`
- State accumulator: `b_h = hl.zeros([dhead, tile_v], dtype=acc_dtype)`
- MatMul: `hl.dot(b_w, c_h)` for w@h and `hl.dot(p_k.T, b_v, acc=b_h)` for k^T@v
- K dimension is specialized: `dhead = hl.specialize(dhead)`

## Full helion.Config Space

### Core knobs
- `block_sizes`: list of tile sizes for each `hl.tile()` call. [batch_tile, head_tile, v_tile, seq_tile]
- `num_warps`: 1, 2, 4, 8, or 16
- `num_stages`: pipeline stages (1-5)

### Indexing mode
- `indexing`: `"pointer"` (default), `"block_ptr"`, or `"tensor_descriptor"`

### Loop & tiling controls
- `flatten_loops`: fuse multi-dimensional tiles into 1D
- `reduction_loops`: roll reductions into loops (e.g., [512] for large K*V)
- `loop_orders`: reorder iteration dimensions
- Per-loop: `range_unroll_factors`, `range_num_stages`, `range_multi_buffers`, `range_flattens`, `range_warp_specializes`

### Parallelism controls
- `pid_type`: `"flat"`, `"xyz"`, `"persistent_interleaved"`, `"persistent_blocked"`
- `l2_grouping`: group PIDs for L2 cache locality

### CompileIQ / ACF
- `advanced_controls_file`: path to ACF file
- Available: `/opt/booster_pack/chunk_fwd_h_0.acf`, `chunk_fwd_h_1.acf`
- 5-15% speedup

## Test & Benchmark Shapes
Tests: (B=1,T=64,H=2,K=64,V=64), (B=2,T=128,H=4,K=64,V=64), (B=1,T=256,H=4,K=64,V=128)
Benchmarks: (B=1,T=64,H=1,K=64,V=64), (B=2,T=512,H=3,K=64,V=64), (B=2,T=1024,H=3,K=64,V=64)

## Optimization Strategy

### Compute-bound kernel
- Two matmuls per chunk: w@h [C,K]@[K,V] and k^T@v [K,C]@[C,V]
- C=64, K=64, V=64 or 128 — fits in registers
- Sequential across chunks but parallel across B*H and V dimension

### Config tuning priorities
1. **block_sizes[V]**: V=64 fits in one block; V=128 may need tiling. Try 64, 128.
2. **num_warps**: 4-8 for matmul-heavy kernels. More warps = more parallelism within the block.
3. **num_stages**: Pipeline stages for the sequential loop. Try 1-3.
4. **indexing**: `"tensor_descriptor"` for B200 TMA.
5. **ACF files**: Try both chunk_fwd_h_0.acf and chunk_fwd_h_1.acf.

### Algorithmic opportunities
- K=64 is specialized → matmul dimensions are compile-time constants
- The inner matmuls (64×64 @ 64×V) are small — fit in registers
- v_new storage and gating can be fused with the state update
- Consider loop unrolling for the sequential chunk loop
- For V=128: tile V into 2 blocks of 64 for better register usage

## Debug & Iteration Workflow
- `HELION_INTERPRET=1` — run in Python interpreter (slow but debuggable)
- `HELION_AUTOTUNE_EFFORT=none` — skip autotuning, use provided config

## Constraints
- Must use Helion DSL (inline Triton/ASM ≤30% LOC)
- Must pass all test shapes (rtol=1e-2, atol=1e-2)
- Config must be hardcoded per shape
