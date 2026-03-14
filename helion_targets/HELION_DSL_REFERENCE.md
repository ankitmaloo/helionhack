# Helion DSL Quick Reference — Verified Patterns

This document captures what WORKS and what DOESN'T in the Helion DSL,
verified against the codebase and tested on B200.

## 1. Kernel Declaration

```python
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, config=config)
def my_kernel(x: torch.Tensor, y: torch.Tensor, param: int) -> torch.Tensor:
    # ...
    return out

# Can also return tuples:
def my_kernel(...) -> tuple[torch.Tensor, torch.Tensor]:
    return out1, out2
```

## 2. hl.tile() — Iteration

```python
# Single dimension:
for tile_m in hl.tile(M):
    ...

# Multiple dimensions (fused):
for tile_m, tile_n in hl.tile([M, N]):
    ...

# Explicit block_size:
for tile_b, tile_h, tile_v in hl.tile([B, H, V], block_size=[1, 1, block_v]):
    ...

# Nested tiles (outer + inner):
for tile_outer in hl.tile([B, H], block_size=[1, 1]):
    for tile_inner in hl.tile(T, block_size=chunk_size):
        ...

# Range-based (start, end):
for tile in hl.tile(start, end, block_size=bs):
    ...
```

**config.block_sizes**: One entry per `hl.register_block_size()` call + one per `hl.tile()`
dimension that doesn't have explicit block_size. Explicit block_size=[1, 1, block_v] means
only block_v is configurable — the 1s are fixed.

## 3. hl.register_block_size() — Tunable Block Size

```python
block_v = hl.register_block_size(V)  # Makes V-dim block size tunable
# Then use in tile:
for tile_v in hl.tile(V, block_size=block_v):
    ...
# Or in combined tile:
for tile_b, tile_v in hl.tile([B, V], block_size=[1, block_v]):
    ...
```

**config.block_sizes** has one entry per `hl.register_block_size()` call.

## 4. hl.specialize() — Compile-Time Constants

```python
D = hl.specialize(D)                    # From tensor shape: x.shape[-1]
chunk_size = hl.specialize(chunk_size)   # From int parameter
W = hl.specialize(weight.shape[1])      # From tensor dim
```

**MUST** be tensor dimension, tensor stride, or kernel int parameter.
**CANNOT** be a module-level global constant — pass as parameter first.

## 5. hl.dot() — Matrix Multiply

```python
# Basic:
result = hl.dot(a, b)                      # [M,K] @ [K,N] → [M,N]

# With accumulator (fused multiply-add):
acc = hl.dot(a, b, acc=acc)                # accumulates into acc

# With output dtype:
result = hl.dot(a, b, out_dtype=torch.float32)

# Transpose:
qk = hl.dot(q, k.T, out_dtype=torch.float32)  # q @ k^T

# Common pattern for matmul reduction:
acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
for tile_k in hl.tile(K):
    acc = hl.dot(x[tile_m, tile_k], y[tile_k, tile_n], acc=acc)
```

## 6. hl.zeros() / hl.full() — Register Tensors

```python
acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
acc = hl.zeros([dhead, tile_v], dtype=torch.float32)  # mixed int + tile
mi = hl.full([tile_m], float("-inf"), dtype=torch.float32)
```

Shape can mix specialized ints and tile variables.

## 7. Tile Properties

```python
tile.id       # tile index (begin // block_size) — SCALAR
tile.begin    # start offset — SCALAR
tile.end      # end offset — SCALAR
tile.index    # 1D tensor of absolute indices — TENSOR
tile.block_size  # block size — SCALAR
tile.count    # total number of tiles — SCALAR

# tile.id usage (for chunk indexing):
for t_i in hl.tile(seqlen, block_size=chunk_size):
    h[i_b, t_i.id, i_h, :, tile_v] = state  # t_i.id is chunk number

# tile.index usage (for masking):
m_t = t_i.index < seqlen  # boolean mask

# tile.begin usage:
t_last = min(t_i.begin + chunk_size, seqlen) - 1
```

**CAVEAT**: `tile.id` may fail inside deeply nested tile loops.
Workaround: reshape data so chunk dimension is explicit, tile over it with block_size=1.

## 8. Tensor Indexing

```python
# Scalar tile (block_size=1) → use .id for scalar index:
i_b = tile_b.id
x[i_b, :, :]           # scalar index + full slices

# Block tile → use directly for block load:
x[tile_m, tile_n]       # loads block [tm, tn]
x[i_b, tile_t, i_h, :] # scalar, block, scalar, full → [block_t, K]

# Shifted access (VERIFIED WORKING):
x_pad[tile_n, tile_s + k]  # offset by constant k → shifted block load

# Full slice with :
x[i_b, t_i, i_h, :]     # loads all elements along last dim
```

## 9. Broadcasting Inside Kernels

```python
# Unsqueeze with None (VERIFIED):
w_k[:, None] * x_k         # [td, 1] * [td, ts] → [td, ts]
b_g[:, None] - b_g[None, :] # [C, 1] - [1, C] → [C, C]
o * torch.exp(g)[:, None]   # [C, tv] * [C, 1] → [C, tv]

# Direct broadcasting:
acc + bias[tile_d][:, None]  # [td, ts] + [td, 1] → [td, ts]
```

## 10. Supported PyTorch Operations in Kernels

**Elementwise**: torch.exp, torch.abs, torch.clamp, torch.where, torch.sigmoid,
torch.sqrt, torch.rsqrt, torch.sin, torch.cos, torch.tanh, torch.relu, +, -, *, /

**Reductions**: torch.amax, torch.amin, torch.sum, torch.mean, torch.all, torch.any

**Comparison**: ==, !=, <, >, <=, >=, &, |

**Creation**: torch.empty, torch.empty_like, torch.zeros, torch.ones,
torch.arange, torch.tril, torch.triu

**Type conversion**: .to(dtype), .float(), .half(), .bfloat16()

## 11. Config Space — Full Reference

```python
helion.Config(
    # Core (ALWAYS set these):
    block_sizes=[64],        # One per tunable tile dim
    num_warps=4,             # 1, 2, 4, 8, or 16
    num_stages=1,            # 1-5 (pipeline stages)

    # Indexing:
    indexing="pointer",      # "pointer", "block_ptr", "tensor_descriptor"

    # Loop controls:
    flatten_loops=False,     # Fuse multi-dim tiles to 1D
    reduction_loops=[],      # Roll reductions into loops
    loop_orders=[],          # Reorder iteration dims

    # Parallelism:
    pid_type="flat",         # "flat", "xyz", "persistent_interleaved", "persistent_blocked"
    l2_grouping=1,           # L2 cache locality grouping

    # Per-loop tunables (lists, one per loop):
    range_unroll_factors=[],
    range_num_stages=[],
    range_multi_buffers=[],
    range_flattens=[],
    range_warp_specializes=[],

    # CompileIQ ACF:
    advanced_controls_file=None,  # Path to .acf file
)
```

## 12. Common Gotchas

1. **block_sizes count**: Must match number of `hl.register_block_size()` calls.
   Explicit block_size=[1, 1, block_v] in hl.tile() means only 1 configurable entry (for block_v).

2. **hl.specialize on globals**: FAILS. Must pass as kernel parameter first.
   ```python
   # BAD:  chunk_size = hl.specialize(CHUNK_SIZE)  # module global
   # GOOD: def kernel(x, chunk_size: int): chunk_size = hl.specialize(chunk_size)
   ```

3. **Rank consistency in loops**: All paths through a loop must produce tensors of same rank.
   Initialize accumulators with hl.zeros() at the correct rank.

4. **tile.id in nested tiles**: May fail with internal assertion. Workaround: reshape data
   to make the indexed dimension explicit, then use scalar .id from the outer tile.

5. **Kernel return**: Can return single tensor or tuple of tensors. Use pre-allocated
   output tensors as parameters for complex multi-output patterns.

6. **Static shapes**: `static_shapes=True` requires fixed shapes across calls with same config.
   Use separate configs per shape in SHAPE_CONFIGS dict.

## 13. Verified Working Kernel Patterns

### Pattern A: Reduction (fp8_quant)
```python
for row_idx in hl.tile(N):
    row = x[row_idx, :].to(torch.float32)
    absmax = torch.amax(torch.abs(row), -1)
    # ... scale and quantize
```

### Pattern B: Sequential Recurrence (gdn_fwd_h)
```python
block_v = hl.register_block_size(dstate)
for tile_b, tile_h, tile_v in hl.tile([B, H, V], block_size=[1, 1, block_v]):
    i_b, i_h = tile_b.id, tile_h.id
    b_h = hl.zeros([dhead, tile_v], dtype=torch.float32)
    for t_i in hl.tile(seqlen, block_size=chunk_size):
        h[i_b, t_i.id, i_h, :, tile_v] = b_h.to(dtype)
        b_w = w[i_b, t_i, i_h, :]
        b_v = hl.dot(b_w, b_h.to(dtype), out_dtype=torch.float32)
        # ... gate and update
        b_h = hl.dot(p_k.T, b_v, acc=b_h)
```

### Pattern C: Attention (chunk_fwd_o)
```python
# Pre-chunk data: [B, NT, C, H, K]
for tile_b, tile_c, tile_h, tile_v in hl.tile([B, NT, H, V], block_size=[1, 1, 1, block_v]):
    i_b, i_c, i_h = tile_b.id, tile_c.id, tile_h.id
    b_q = q[i_b, i_c, :, i_h, :]     # [C, K]
    qk = hl.dot(b_q, b_k.T)          # [C, C]
    mask = torch.arange(C)[:, None] >= torch.arange(C)[None, :]
    qk = qk * torch.where(mask, torch.exp(g_diff), torch.zeros_like(g_diff))
    out_val = hl.dot(qk, b_v)         # [C, V]
```

### Pattern D: Shifted Depthwise Conv
```python
for tile_n in hl.tile(N):
    for tile_s in hl.tile(S):
        acc = hl.zeros([tile_n, tile_s], dtype=torch.float32)
        for k in range(W):  # W is specialized
            w_k = weight[tile_n, k]
            x_k = x_pad[tile_n, tile_s + k]  # shifted access
            acc = acc + w_k[:, None] * x_k
```

### Pattern E: Chunk Matmul (recompute_w_u)
```python
block_d = hl.register_block_size(D)
for tile_b, tile_h, tile_d in hl.tile([B, H, D], block_size=[1, 1, block_d]):
    i_b, i_h = tile_b.id, tile_h.id
    for t_i in hl.tile(T, block_size=chunk_size):
        b_A = A[i_b, t_i, i_h, :]     # [BT, BT]
        b_x = x[i_b, t_i, i_h, tile_d] # [BT, td]
        out[i_b, t_i, i_h, tile_d] = hl.dot(b_A, b_x)
```
