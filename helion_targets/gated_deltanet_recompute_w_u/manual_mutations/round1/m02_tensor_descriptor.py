from task import input_t, output_t

import torch
import helion
import helion.language as hl


CHUNK_SIZE = 64


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def chunk_matmul_kernel(
        A: torch.Tensor,
        x_scaled: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        batch, seqlen, nheads, dout = x_scaled.shape
        chunk_size = hl.specialize(chunk_size)
        hl.specialize(dout)

        acc_dtype = torch.float32
        dtype = x_scaled.dtype

        out = torch.empty_like(x_scaled)
        block_d = hl.register_block_size(dout)

        for tile_b, tile_h, tile_d in hl.tile(
            [batch, nheads, dout], block_size=[1, 1, block_d]
        ):
            i_b = tile_b.id
            i_h = tile_h.id
            for t_i in hl.tile(seqlen, block_size=chunk_size):
                b_A = A[i_b, t_i, i_h, :].to(acc_dtype)
                b_x = x_scaled[i_b, t_i, i_h, tile_d].to(acc_dtype)
                b_out = hl.dot(b_A, b_x, out_dtype=acc_dtype)
                out[i_b, t_i, i_h, tile_d] = b_out.to(dtype)

        return out

    return chunk_matmul_kernel


SHAPE_CONFIGS: dict[tuple, tuple[helion.Config, helion.Config]] = {
    (1, 64, 2, 64, 64): (
        helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=4, num_stages=1),
        helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=4, num_stages=1),
    ),
    (2, 128, 4, 64, 64): (
        helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=4, num_stages=1),
        helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=4, num_stages=1),
    ),
    (1, 256, 4, 64, 128): (
        helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=4, num_stages=2),
        helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=8, num_stages=2),
    ),
    (1, 64, 1, 64, 64): (
        helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=4, num_stages=2),
        helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=4, num_stages=2),
    ),
    (2, 512, 3, 64, 64): (
        helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=8, num_stages=2),
        helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=8, num_stages=2),
    ),
    (2, 1024, 3, 64, 64): (
        helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=8, num_stages=3),
        helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=8, num_stages=3),
    ),
}


_KERNELS = {
    shape: (_make_kernel(cfg_k), _make_kernel(cfg_v))
    for shape, (cfg_k, cfg_v) in SHAPE_CONFIGS.items()
}


def custom_kernel(data: input_t) -> output_t:
    k, v, beta, A, g = data
    B, T, H, K = k.shape
    V = v.shape[-1]

    kernel_w, kernel_u = _KERNELS[(B, T, H, K, V)]

    v_scaled = v * beta.unsqueeze(-1)
    u = kernel_u(A, v_scaled, CHUNK_SIZE)

    k_scaled = k * (beta * torch.exp(g)).unsqueeze(-1)
    w = kernel_w(A, k_scaled, CHUNK_SIZE)

    return w, u
