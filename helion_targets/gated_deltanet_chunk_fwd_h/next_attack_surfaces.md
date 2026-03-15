# Gated DeltaNet chunk_fwd_h — Next Attack Surfaces

Based on the failures of the structural rewrites in Round 3 and the identified set-level gaps, here are the next high-value attack surfaces to explore for improving kernel performance beyond the `31.2 us` baseline wall.

## 1. Math and Precision Reordering
The inner loop performs a sequence of subtractions, exponential scalings, and dtype casts. 
- **Cast Minimization**: `b_v` and `b_h` are converted back and forth between `acc_dtype` and `dtype`. We can delay these casts or keep `b_h_store` bound closer to the math ops.
- **Exponential Factorization**: Mathematically rewrite `exp(g_last - g)` to avoid computing the subtraction dynamically per element, or precalculate the sequence offsets.
- **In-place Operations**: Explicitly force in-place updates (e.g., `b_h.add_()`, `b_v.mul_()`) to guide the compiler away from allocating new registers for intermediate value flows.

## 2. Asymmetric Memory Access Indexing
The tensor `g` is small `[B, T, H]` compared to the main `k, w, u` tensors `[B, T, H, D]`. 
- **Split Indexing Strategies**: Continue using `tensor_descriptor` (TMA) for the massive `k, w, u` chunks, but try `block_ptr` or direct vectorized pointer arithmetic for `g` to avoid TMA setup overhead on small reads.
- **Warp-level Broadcasts**: Load `g` slices via warp-level broadcasts since it lacks a `D` dimension, reducing L2 cache contention.

## 3. Explicit Instruction Overlap and Memory Hiding
While `num_stages` allows the compiler to pipeline, manual layout can force better overlapping.
- **Delayed Write-back**: The state write (`h_out[...] = b_h_store`) happens early in the chunk loop. Moving it down to overlap with the compute of `b_v` can hide memory store latency.
- **Fusing Dot-Accumulation**: Modify the final update `b_h = hl.dot(..., acc=b_h)` to happen synchronously with the next loop's `b_w` fetch.

## 4. Retuning Schedules for Structural Changes
Many of the round 3 rewrites (like chunked sequential) were evaluated under the baseline schedule (`num_warps=8, num_stages=3`).
- **Low-Register Configs**: The explicit views and value-flow splits likely spilled registers. We should test these specific rewrites with `num_warps=4` and `num_stages=2` to trade occupancy for register space.
- **Dynamic V-Tiling**: `block_v` is registered globally. Modifying `num_warps` to slice `V` differently per benchmark shape might uncover better multiprocessor utilization.

## 5. Control Flow Flattening
- **Flattening the Loop Structure**: Instead of nesting `tile_v` outside and `t_i` inside, we can try exploring reversing the loop hierarchy or using a single flat loop over `T` and `V` if the dependencies allow, to reduce branch divergence.
