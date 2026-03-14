from task import input_t, output_t

import torch
import helion
import helion.language as hl


CHUNK_SIZE = 64


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def chunk_fwd_h_kernel(
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor,
        v_new_out: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        batch, seqlen, nheads, dhead = k.shape
        dhead = hl.specialize(dhead)
        chunk_size = hl.specialize(chunk_size)
        dstate = u.shape[-1]

        acc_dtype = torch.float32
        dtype = k.dtype

        nchunks = seqlen // chunk_size
        h_out = torch.empty(batch, nchunks, nheads, dhead, dstate, dtype=dtype, device=k.device)
        block_v = hl.register_block_size(dstate)

        for tile_b, tile_h, tile_v in hl.tile([batch, nheads, dstate], block_size=[1, 1, block_v]):
            i_b = tile_b.id
            i_h = tile_h.id
            b_h = hl.zeros([dhead, tile_v], dtype=acc_dtype)
            for t_i in hl.tile(seqlen, block_size=chunk_size):
                b_h_store = b_h.to(dtype)
                h_out[i_b, t_i.id, i_h, :, tile_v] = b_h_store
                b_w = w[i_b, t_i, i_h, :]
                b_v = hl.dot(b_w, b_h_store, out_dtype=acc_dtype)
                p_v = u[i_b, t_i, i_h, tile_v].to(acc_dtype)
                b_v = p_v - b_v
                v_new_out[i_b, t_i, i_h, tile_v] = b_v.to(dtype)
                t_i_last = t_i.begin + chunk_size - 1
                b_g_last = g[i_b, t_i_last, i_h].to(acc_dtype)
                b_g = g[i_b, t_i, i_h].to(acc_dtype)
                b_v *= torch.exp(b_g_last - b_g)[:, None]
                b_h *= torch.exp(b_g_last)
                p_k = k[i_b, t_i, i_h, :]
                b_h = hl.dot(p_k.T, b_v.to(dtype), acc=b_h)

        return h_out

    return chunk_fwd_h_kernel


SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=2),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=8, num_stages=2),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[64], indexing="tensor_descriptor", num_warps=8, num_stages=3),
}


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    v_new = torch.empty(B, T, H, V, dtype=u.dtype, device=u.device)
    h = kernel(k, w, u, g, v_new, CHUNK_SIZE)
    return h, v_new
