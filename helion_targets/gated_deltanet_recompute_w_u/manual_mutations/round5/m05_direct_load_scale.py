from task import input_t, output_t

import torch
import helion
import helion.language as hl


CHUNK_SIZE = 64


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def fused_direct_kernel(
        A: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: torch.Tensor,
        gate_scale: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seqlen, nheads, dk = k.shape
        dv = v.shape[-1]
        chunk_size = hl.specialize(chunk_size)
        hl.specialize(dk)
        hl.specialize(dv)
        acc_dtype = torch.float32
        dtype = k.dtype
        out_w = torch.empty_like(k)
        out_u = torch.empty_like(v)
        block_k = hl.register_block_size(dk)
        block_v = hl.register_block_size(dv)
        for tile_b, tile_t, tile_h in hl.tile([batch, seqlen, nheads], block_size=[1, chunk_size, 1]):
            i_b = tile_b.id
            i_h = tile_h.id
            b_A = A[i_b, tile_t, i_h, :].to(acc_dtype)
            beta_t = beta[i_b, tile_t, i_h].to(acc_dtype)[:, None]
            gate_t = gate_scale[i_b, tile_t, i_h].to(acc_dtype)[:, None]
            for tile_k in hl.tile(dk, block_size=block_k):
                out_w[i_b, tile_t, i_h, tile_k] = hl.dot(
                    b_A, k[i_b, tile_t, i_h, tile_k].to(acc_dtype) * gate_t, out_dtype=acc_dtype
                ).to(dtype)
            for tile_v in hl.tile(dv, block_size=block_v):
                out_u[i_b, tile_t, i_h, tile_v] = hl.dot(
                    b_A, v[i_b, tile_t, i_h, tile_v].to(acc_dtype) * beta_t, out_dtype=acc_dtype
                ).to(dtype)
        return out_w, out_u

    return fused_direct_kernel


SHAPE_CONFIGS = {
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=1),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=1),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=2),
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[32, 32], num_warps=8, num_stages=2),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[32, 32], num_warps=8, num_stages=3),
}


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    gate_scale = beta * torch.exp(g)
    return kernel(A, k, v, beta, gate_scale, CHUNK_SIZE)
