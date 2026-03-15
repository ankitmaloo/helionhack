# Gated DeltaNet chunk_fwd_h — Manual Mutation Iteration 04 (Big Swings)

## Baseline
Target: `helion_targets/gated_deltanet_chunk_fwd_h/seed.py` (which currently contains the `m01_descriptor_no_mask` winner from Round 3).
Current best runtime: `31.2 us`.

## 10 Attack Hypotheses (Based on Next Attack Surfaces)
1. **Delayed Write-back**: Move the state write (`h_out[...] = b_h_store`) to happen later in the loop (e.g., after computing `b_v`) to overlap memory store latency with math instructions.
2. **In-place Math**: Force register reuse by using in-place operations (`b_v.mul_()`, `b_h.mul_()`) instead of out-of-place assignments.
3. **Exponential Factorization**: Compute `g_diff = b_g_last - b_g` first, then a single `exp`, rather than evaluating complex expressions inline.
4. **Lighter Schedule for Descriptor**: The descriptor config uses `num_warps=8, num_stages=3`. Drop it to `num_warps=4, num_stages=2` to trade occupancy for register space, mitigating spilling.
5. **Warp-level Broadcasts for `g`**: Explicitly load `g` as a 1D tile and broadcast it manually to avoid implicit broadcasting overhead during the `b_v` scaling.
6. **Reversed Loop Order**: Flip `tile_v` and `t_i` if possible, though sequence dependence makes this invalid.
7. **Delayed Casting**: Push `b_v.to(dtype)` directly into the `hl.dot` call instead of doing it on a separate line, reducing the lifetime of the converted variable.
8. **Combined Delayed Write-back & In-place**: Combine the memory hiding of H1 with the register efficiency of H2.
9. **Fused State Update**: Keep `b_h` in `dtype` entirely to avoid the `to(dtype)` overhead, accepting potential precision loss (though correctness might fail).
10. **Pre-load `g` chunk**: Load the entire `g` chunk for the sequence tile at the very beginning of the loop to ensure it's in registers before the heavy `hl.dot` starts.

## 5 Selected Mutations for Round 4
1. `m01_delayed_writeback.py`: Moves `h_out` store to overlap with `b_v` compute.
2. `m02_inplace_math.py`: Replaces variable reassignments with `.mul_()` and explicit in-place arithmetic.
3. `m03_exponential_factorization.py`: Simplifies the `g` math block by pre-calculating the difference and explicit scaling factors.
4. `m04_lighter_schedule.py`: Changes the large shape configs from `num_warps=8, num_stages=3/2` to `num_warps=4, num_stages=2` to reduce register spilling.
5. `m05_combined_writeback_inplace.py`: Combines delayed writeback with in-place math for maximum register/instruction efficiency.

## Run Objective
Evaluate these 5 distinct "big swing" ideas in parallel against the newly promoted baseline to break the `31.2 us` wall.
