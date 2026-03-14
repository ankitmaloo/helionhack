from task import input_t, output_t

import torch
import helion
import helion.language as hl


CHUNK_SIZE = 64


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def fused_chunk_kernel(
        A_c: torch.Tensor,
        k_c: torch.Tensor,
        v_c: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, nchunks, nheads, _, dk = k_c.shape
        dv = v_c.shape[-1]
        chunk_size = hl.specialize(chunk_size)
        hl.specialize(dk)
        hl.specialize(dv)
        acc_dtype = torch.float32
        dtype = k_c.dtype
        out_w = torch.empty_like(k_c)
        out_u = torch.empty_like(v_c)
        block_k = hl.register_block_size(dk)
        block_v = hl.register_block_size(dv)

        for tile_b, tile_c, tile_h in hl.tile([batch, nchunks, nheads], block_size=[1, 1, 1]):
            i_b = tile_b.id
            i_c = tile_c.id
            i_h = tile_h.id
            b_A = A_c[i_b, i_c, i_h, :, :].to(acc_dtype)

            for tile_k in hl.tile(dk, block_size=block_k):
                out_w[i_b, i_c, i_h, :, tile_k] = hl.dot(
                    b_A,
                    k_c[i_b, i_c, i_h, :, tile_k].to(acc_dtype),
                    out_dtype=acc_dtype,
                ).to(dtype)

            for tile_v in hl.tile(dv, block_size=block_v):
                out_u[i_b, i_c, i_h, :, tile_v] = hl.dot(
                    b_A,
                    v_c[i_b, i_c, i_h, :, tile_v].to(acc_dtype),
                    out_dtype=acc_dtype,
                ).to(dtype)

        return out_w, out_u

    return fused_chunk_kernel


SHAPE_CONFIGS = {
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=1),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=1),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=2),
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[32, 32], num_warps=8, num_stages=2),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[32, 32], num_warps=8, num_stages=3),
}


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def _chunk_a(A: torch.Tensor, chunk_size: int) -> torch.Tensor:
    B, T, H, _ = A.shape
    nchunks = T // chunk_size
    return A.reshape(B, nchunks, chunk_size, H, chunk_size).permute(0, 1, 3, 2, 4).contiguous()


def _chunk_x(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    B, T, H, D = x.shape
    nchunks = T // chunk_size
    return x.reshape(B, nchunks, chunk_size, H, D).permute(0, 1, 3, 2, 4).contiguous()


def _unchunk_x(x_c: torch.Tensor) -> torch.Tensor:
    B, nchunks, H, chunk_size, D = x_c.shape
    return x_c.permute(0, 1, 3, 2, 4).reshape(B, nchunks * chunk_size, H, D)


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    A_c = _chunk_a(A, CHUNK_SIZE)
    v_c = _chunk_x(v * beta.unsqueeze(-1), CHUNK_SIZE)
    k_c = _chunk_x(k * (beta * torch.exp(g)).unsqueeze(-1), CHUNK_SIZE)
    w_c, u_c = kernel(A_c, k_c, v_c, CHUNK_SIZE)
    return _unchunk_x(w_c), _unchunk_x(u_c)
