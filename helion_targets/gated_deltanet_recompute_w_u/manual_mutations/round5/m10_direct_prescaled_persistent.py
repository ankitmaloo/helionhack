from task import input_t, output_t

import torch
import helion
import helion.language as hl


CHUNK_SIZE = 64


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def fused_direct_kernel(
        A: torch.Tensor,
        k_scaled: torch.Tensor,
        v_scaled: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seqlen, nheads, dk = k_scaled.shape
        dv = v_scaled.shape[-1]
        chunk_size = hl.specialize(chunk_size)
        hl.specialize(dk)
        hl.specialize(dv)
        acc_dtype = torch.float32
        dtype = k_scaled.dtype
        out_w = torch.empty_like(k_scaled)
        out_u = torch.empty_like(v_scaled)
        block_k = hl.register_block_size(dk)
        block_v = hl.register_block_size(dv)
        for tile_b, tile_t, tile_h in hl.tile([batch, seqlen, nheads], block_size=[1, chunk_size, 1]):
            i_b = tile_b.id
            i_h = tile_h.id
            b_A = A[i_b, tile_t, i_h, :].to(acc_dtype)
            for tile_k in hl.tile(dk, block_size=block_k):
                out_w[i_b, tile_t, i_h, tile_k] = hl.dot(
                    b_A, k_scaled[i_b, tile_t, i_h, tile_k].to(acc_dtype), out_dtype=acc_dtype
                ).to(dtype)
            for tile_v in hl.tile(dv, block_size=block_v):
                out_u[i_b, tile_t, i_h, tile_v] = hl.dot(
                    b_A, v_scaled[i_b, tile_t, i_h, tile_v].to(acc_dtype), out_dtype=acc_dtype
                ).to(dtype)
        return out_w, out_u

    return fused_direct_kernel


SHAPE_CONFIGS = {
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=1),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=1),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=2),
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[32, 32], pid_type="persistent_interleaved", num_warps=8, num_stages=2),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[32, 32], pid_type="persistent_interleaved", num_warps=8, num_stages=3),
}


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    v_scaled = v * beta.unsqueeze(-1)
    k_scaled = k * (beta * torch.exp(g)).unsqueeze(-1)
    return kernel(A, k_scaled, v_scaled, CHUNK_SIZE)
