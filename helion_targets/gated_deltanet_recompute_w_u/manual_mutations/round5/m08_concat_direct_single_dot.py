from task import input_t, output_t

import torch
import helion
import helion.language as hl


CHUNK_SIZE = 64


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def concat_direct_kernel(A: torch.Tensor, x_cat: torch.Tensor, chunk_size: int) -> torch.Tensor:
        batch, seqlen, nheads, dcat = x_cat.shape
        chunk_size = hl.specialize(chunk_size)
        hl.specialize(dcat)
        acc_dtype = torch.float32
        dtype = x_cat.dtype
        out = torch.empty_like(x_cat)
        block_d = hl.register_block_size(dcat)
        for tile_b, tile_t, tile_h in hl.tile([batch, seqlen, nheads], block_size=[1, chunk_size, 1]):
            i_b = tile_b.id
            i_h = tile_h.id
            b_A = A[i_b, tile_t, i_h, :].to(acc_dtype)
            for tile_d in hl.tile(dcat, block_size=block_d):
                out[i_b, tile_t, i_h, tile_d] = hl.dot(
                    b_A, x_cat[i_b, tile_t, i_h, tile_d].to(acc_dtype), out_dtype=acc_dtype
                ).to(dtype)
        return out

    return concat_direct_kernel


SHAPE_CONFIGS = {
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[128], num_warps=8, num_stages=2),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[128], num_warps=8, num_stages=3),
}


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    v_scaled = v * beta.unsqueeze(-1)
    k_scaled = k * (beta * torch.exp(g)).unsqueeze(-1)
    out_cat = kernel(A, torch.cat([k_scaled, v_scaled], dim=-1), CHUNK_SIZE)
    return out_cat[..., :K], out_cat[..., K:]
