# Gated DeltaNet chunk_fwd_h Skill — Sequential Recurrence and V-Tiling Layer

## Objective
Improve `gated_deltanet_chunk_fwd_h` by reducing benchmark runtime on B200 while preserving correctness for all test shapes.

## Selected code layer
The primary layer for the first mutation round is the recurrence schedule and V-tiling layer.

This includes:
- `block_sizes` for the V/state dimension
- `num_warps`
- `num_stages`
- `indexing`
- `advanced_controls_file`
- the sequential chunk traversal structure

## Why this layer first
- The kernel is sequential across chunks, so outer scheduling matters a lot.
- Each chunk performs two small matmuls that should fit well in registers.
- The baseline uses one uniform config for every shape, which is unlikely to be optimal.
- Structural chunk handling may unlock cleaner memory access without changing the recurrence semantics.

## Invariants
These must remain true:
- chunks are processed in order
- `h_out` stores the state before each chunk is processed
- `v_new_out` stores ungated corrected values before the gated state update
- the output shapes remain `(h, v_new)` with `h` as `[B, NT, H, K, V]` and `v_new` as `[B, T, H, V]`
- the kernel remains in Helion DSL

## Safe mutation surface
Low-risk edits:
- per-shape `num_warps` and `num_stages`
- benchmark-only config changes
- large-shape-only descriptor or ACF use
- explicit chunked input layout while keeping sequential chunk processing
- moderate V-tiling changes for `V=128`

## High-risk edits to avoid in round 1
- removing the sequential dependency across chunks
- changing the mathematical order of `v_new`, gating, and state update
- mixing multiple structural rewrites in one candidate
- changing public inputs or outputs

## Likely performance levers
- deeper pipeline for larger benchmark shapes
- stronger warp count for the 512 and 1024 benchmarks
- explicit chunk layout to avoid repeated strided chunk indexing
- large-shape-only descriptor or ACF guidance

## Measurement protocol
For every mutation, track:
- validity
- correctness
- mean and minimum runtime
- test and benchmark times
- failure reasons

## Baseline status
The baseline seed currently uses one conservative config family:
- `block_sizes=[64]`
- `num_warps=4`
- `num_stages=1`

Manual round-1 baseline should be re-measured before ranking candidates.
