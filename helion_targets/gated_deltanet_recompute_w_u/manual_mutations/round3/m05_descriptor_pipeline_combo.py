from task import input_t, output_t

import torch
import helion
import helion.language as hl


CHUNK_SIZE = 64


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def chunk_matmul_kernel(A_c: torch.Tensor, x_c: torch.Tensor, chunk_size: int) -> torch.Tensor:
        batch, nchunks, nheads, _, dout = x_c.shape
        chunk_size = hl.specialize(chunk_size)
        hl.specialize(dout)
        acc_dtype = torch.float32
        dtype = x_c.dtype
        out = torch.empty_like(x_c)
        block_d = hl.register_block_size(dout)
        for tile_b, tile_c, tile_h, tile_d in hl.tile([batch, nchunks, nheads, dout], block_size=[1, 1, 1, block_d]):
            i_b = tile_b.id
            i_c = tile_c.id
            i_h = tile_h.id
            out[i_b, i_c, i_h, :, tile_d] = hl.dot(
                A_c[i_b, i_c, i_h, :, :].to(acc_dtype),
                x_c[i_b, i_c, i_h, :, tile_d].to(acc_dtype),
                out_dtype=acc_dtype,
            ).to(dtype)
        return out
    return chunk_matmul_kernel


SHAPE_CONFIGS = {
    (1, 64, 2, 64, 64): (
        helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
        helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    ),
    (2, 128, 4, 64, 64): (
        helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
        helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    ),
    (1, 256, 4, 64, 128): (
        helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
        helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    ),
    (1, 64, 1, 64, 64): (
        helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
        helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    ),
    (2, 512, 3, 64, 64): (
        helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=8, num_stages=3),
        helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=8, num_stages=3),
    ),
    (2, 1024, 3, 64, 64): (
        helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=8, num_stages=4),
        helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=8, num_stages=4),
    ),
}


_KERNELS = {shape: (_make_kernel(cfg_k), _make_kernel(cfg_v)) for shape, (cfg_k, cfg_v) in SHAPE_CONFIGS.items()}


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
    kernel_w, kernel_u = _KERNELS[(B, T, H, K, V)]
    A_c = _chunk_a(A, CHUNK_SIZE)
    v_c = _chunk_x(v * beta.unsqueeze(-1), CHUNK_SIZE)
    k_c = _chunk_x(k * (beta * torch.exp(g)).unsqueeze(-1), CHUNK_SIZE)
    u = _unchunk_x(kernel_u(A_c, v_c, CHUNK_SIZE))
    w = _unchunk_x(kernel_w(A_c, k_c, CHUNK_SIZE))
    return w, u
