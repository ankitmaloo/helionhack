from task import input_t, output_t

import torch
import helion
import helion.language as hl


CHUNK_SIZE = 64


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def chunk_fwd_h_kernel(
        k_c: torch.Tensor,
        w_c: torch.Tensor,
        u_c: torch.Tensor,
        g: torch.Tensor,
        g_c: torch.Tensor,
        v_new_out_c: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        batch, nchunks, nheads, _, dhead = k_c.shape
        dhead = hl.specialize(dhead)
        chunk_size = hl.specialize(chunk_size)
        dstate = u_c.shape[-1]

        acc_dtype = torch.float32
        dtype = k_c.dtype

        h_out = torch.empty(batch, nchunks, nheads, dhead, dstate, dtype=dtype, device=k_c.device)
        block_v = hl.register_block_size(dstate)

        for tile_b, tile_h, tile_v in hl.tile([batch, nheads, dstate], block_size=[1, 1, block_v]):
            i_b = tile_b.id
            i_h = tile_h.id
            b_h = hl.zeros([dhead, tile_v], dtype=acc_dtype)
            for c_i in hl.tile(nchunks, block_size=1):
                i_c = c_i.id
                k_chunk = k_c[i_b, i_c, i_h, :, :]
                w_chunk = w_c[i_b, i_c, i_h, :, :]
                u_chunk = u_c[i_b, i_c, i_h, :, tile_v].to(acc_dtype)
                g_chunk = g_c[i_b, i_c, i_h, :].to(acc_dtype)
                last_idx = i_c * chunk_size + chunk_size - 1
                g_last = g[i_b, last_idx, i_h].to(acc_dtype)
                b_h_store = b_h.to(dtype)
                h_out[i_b, i_c, i_h, :, tile_v] = b_h_store
                b_v_raw = u_chunk - hl.dot(w_chunk, b_h_store, out_dtype=acc_dtype)
                v_new_out_c[i_b, i_c, i_h, :, tile_v] = b_v_raw.to(dtype)
                b_h *= torch.exp(g_last)
                b_v_gated = b_v_raw * torch.exp(g_last - g_chunk)[:, None]
                b_h = hl.dot(k_chunk.T, b_v_gated.to(dtype), acc=b_h)

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


def _chunk_tensor(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    B, T, H, D = x.shape
    nchunks = T // chunk_size
    return x.reshape(B, nchunks, chunk_size, H, D).permute(0, 1, 3, 2, 4).contiguous()


def _chunk_gate(g: torch.Tensor, chunk_size: int) -> torch.Tensor:
    B, T, H = g.shape
    nchunks = T // chunk_size
    return g.reshape(B, nchunks, chunk_size, H).permute(0, 1, 3, 2).contiguous()


def _unchunk_v(x_c: torch.Tensor) -> torch.Tensor:
    B, nchunks, H, chunk_size, D = x_c.shape
    return x_c.permute(0, 1, 3, 2, 4).reshape(B, nchunks * chunk_size, H, D)


def custom_kernel(data: input_t) -> output_t:
    k, w, u, g = data
    B, T, H, K = k.shape
    V = u.shape[-1]
    kernel = _KERNELS[(B, T, H, K, V)]
    k_c = _chunk_tensor(k, CHUNK_SIZE)
    w_c = _chunk_tensor(w, CHUNK_SIZE)
    u_c = _chunk_tensor(u, CHUNK_SIZE)
    g_c = _chunk_gate(g, CHUNK_SIZE)
    v_new_c = torch.empty_like(u_c)
    h = kernel(k_c, w_c, u_c, g, g_c, v_new_c, CHUNK_SIZE)
    v_new = _unchunk_v(v_new_c)
    return h, v_new
