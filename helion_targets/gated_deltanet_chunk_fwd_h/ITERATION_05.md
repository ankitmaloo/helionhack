# Gated DeltaNet chunk_fwd_h — Manual Mutation Iteration 05 (Pushing The Big Swing)

## Round 4 Summary
Baseline: `31.2 us`
- `m01_delayed_writeback`: `31.26 us`
- `m02_inplace_math`: `31.20 us`
- **`m03_exponential_factorization`: `31.16 us` (New Winner)**
- `m04_lighter_schedule`: `31.36 us`
- `m05_combined_writeback_inplace`: `31.23 us`

The exponential factorization (`m03`) managed to consistently squeak slightly ahead of the baseline by pre-computing `g_diff = b_g_last - b_g` and scaling cleanly.

## Hypothesis for Round 5 (Going further down the Math / Memory path)
Since we've hit a heavy memory/compute wall around `31.1 us`, we need more aggressive combinations of the winning principles.

1. **Combined Math Path (Factorization + In-place + Delayed Writeback)**: Stack the exact three ideas from Round 4 that didn't crash and showed promise.
2. **Delayed Casting (Late to_dtype)**: Keep `b_v` in `acc_dtype` (FP32) as long as possible. Only cast it right as it goes into the final `hl.dot`. This saves an intermediate cast that gets immediately used.
3. **Explicit Vectorized Load for `g`**: Instead of `g[i_b, t_i, i_h]`, try loading the full `chunk_size` slice into a tensor first at the top of the sequence loop to guarantee it resides in fast registers for the whole block.
4. **Schedule Bump on the Factorized Winner**: Try pushing the warps to 16 for the 1024 shape, or dropping stages back to 1, specifically on the `m03` math logic, since it altered the register lifecycle.
5. **Dtype Homogenization**: The `u` tensor is loaded in `acc_dtype` and math is done there. What if we do the math strictly in `dtype` (BF16/FP16) and only upcast for the matmuls? (Might break correctness, but worth testing the bounds).

## Selected Mutations for Round 5
1. `m01_math_stack_all.py`: Combines exponential factorization, in-place scaling, and delayed write-back.
2. `m02_delayed_casting.py`: Moves the `.to(dtype)` cast on `b_v` strictly into the final `hl.dot` call.
3. `m03_vectorized_g_load.py`: Tries to load `g_chunk = g[i_b, t_i.begin : t_i.begin + chunk_size, i_h]` once, avoiding scalar loads per element.
4. `m04_aggressive_schedule_1024.py`: Bumps `num_warps=16` on the largest shape using the `m03` factorized body.
5. `m05_dtype_homogenization.py`: Converts intermediate `p_v` and `b_v` logic to stay in `dtype` instead of `acc_dtype`.
