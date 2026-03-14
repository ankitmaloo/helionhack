from task import input_t, output_t

import torch
import helion
import helion.language as hl


CHUNK_SIZE = 64


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def chunk_matmul_kernel(A_l: torch.Tensor, x_l: torch.Tensor, chunk_size: int) -> torch.Tensor:
        batch, nh, _, dout = x_l.shape
        chunk_size = hl.specialize(chunk_size)
        hl.specialize(dout)
        acc_dtype = torch.float32
        dtype = x_l.dtype
        out = torch.empty_like(x_l)
        block_d = hl.register_block_size(dout)
        for tile_b, tile_nh, tile_d in hl.tile([batch, nh, dout], block_size=[1, 1, block_d]):
            i_b = tile_b.id
            i_nh = tile_nh.id
            out[i_b, i_nh, :, tile_d] = hl.dot(
                A_l[i_b, i_nh, :, :].to(acc_dtype),
                x_l[i_b, i_nh, :, tile_d].to(acc_dtype),
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
        helion.Config(block_sizes=[32], num_warps=8, num_stages=2),
        helion.Config(block_sizes=[32], num_warps=8, num_stages=2),
    ),
    (2, 1024, 3, 64, 64): (
        helion.Config(block_sizes=[32], num_warps=8, num_stages=3),
        helion.Config(block_sizes=[32], num_warps=8, num_stages=3),
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


def _linearize_a(A_c: torch.Tensor) -> torch.Tensor:
    B, nchunks, H, C, _ = A_c.shape
    return A_c.reshape(B, nchunks * H, C, C)


def _linearize_x(x_c: torch.Tensor) -> torch.Tensor:
    B, nchunks, H, C, D = x_c.shape
    return x_c.reshape(B, nchunks * H, C, D)


def _restore_x(x_l: torch.Tensor, nchunks: int, H: int) -> torch.Tensor:
    B, _, C, D = x_l.shape
    return x_l.reshape(B, nchunks, H, C, D).permute(0, 1, 3, 2, 4).reshape(B, nchunks * C, H, D)


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]
    nchunks = T // CHUNK_SIZE
    kernel_w, kernel_u = _KERNELS[(B, T, H, K, V)]
    A_l = _linearize_a(_chunk_a(A, CHUNK_SIZE))
    v_l = _linearize_x(_chunk_x(v * beta.unsqueeze(-1), CHUNK_SIZE))
    k_l = _linearize_x(_chunk_x(k * (beta * torch.exp(g)).unsqueeze(-1), CHUNK_SIZE))
    u = _restore_x(kernel_u(A_l, v_l, CHUNK_SIZE), nchunks, H)
    w = _restore_x(kernel_w(A_l, k_l, CHUNK_SIZE), nchunks, H)
    return w, u
