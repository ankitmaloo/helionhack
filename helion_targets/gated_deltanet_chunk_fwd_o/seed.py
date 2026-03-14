from task import input_t, output_t

import torch
import helion
import helion.language as hl


CHUNK_SIZE = 64


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def chunk_fwd_o_kernel(
        q_c: torch.Tensor,       # [B, NT, C, H, K] — chunked queries
        k_c: torch.Tensor,       # [B, NT, C, H, K] — chunked keys
        v_c: torch.Tensor,       # [B, NT, C, H, V] — chunked corrected values
        h: torch.Tensor,         # [B, NT, H, K, V] — per-chunk states
        g_c: torch.Tensor,       # [B, NT, C, H] — chunked gates
        scale: float,
        chunk_size: int,
    ) -> torch.Tensor:
        batch, nchunks, _, nheads, dhead = q_c.shape
        dhead = hl.specialize(dhead)
        chunk_size = hl.specialize(chunk_size)
        dstate = v_c.shape[-1]

        acc_dtype = torch.float32
        dtype = q_c.dtype

        out = torch.empty(batch, nchunks, chunk_size, nheads, dstate, dtype=dtype, device=q_c.device)
        block_v = hl.register_block_size(dstate)

        # EVOLVE-BLOCK-START
        for tile_b, tile_c, tile_h, tile_v in hl.tile(
            [batch, nchunks, nheads, dstate], block_size=[1, 1, 1, block_v]
        ):
            i_b = tile_b.id
            i_c = tile_c.id
            i_h = tile_h.id

            # Inter-chunk: q @ h * exp(g)
            b_q = q_c[i_b, i_c, :, i_h, :].to(acc_dtype)       # [C, K]
            b_h = h[i_b, i_c, i_h, :, tile_v].to(acc_dtype)     # [K, tv]
            b_g = g_c[i_b, i_c, :, i_h].to(acc_dtype)           # [C]
            o_inter = hl.dot(b_q, b_h, out_dtype=acc_dtype)      # [C, tv]
            o_inter = o_inter * torch.exp(b_g)[:, None]

            # Intra-chunk: causal QK attention
            b_k = k_c[i_b, i_c, :, i_h, :].to(acc_dtype)       # [C, K]
            b_v = v_c[i_b, i_c, :, i_h, tile_v].to(acc_dtype)   # [C, tv]

            # QK scores
            qk = hl.dot(b_q, b_k.T, out_dtype=acc_dtype)        # [C, C]

            # Causal mask and gating
            g_diff = b_g[:, None] - b_g[None, :]                 # [C, C]
            mask_idx = torch.arange(chunk_size, device=q_c.device)
            causal = mask_idx[:, None] >= mask_idx[None, :]      # [C, C]
            qk = qk * torch.where(causal, torch.exp(g_diff), torch.zeros_like(g_diff))

            # Attention output
            o_intra = hl.dot(qk.to(dtype), b_v, out_dtype=acc_dtype)  # [C, tv]

            out[i_b, i_c, :, i_h, tile_v] = ((o_inter + o_intra) * scale).to(dtype)
        # EVOLVE-BLOCK-END

        return out

    return chunk_fwd_o_kernel


# EVOLVE-BLOCK-START
# Per-shape configs: (B, T, H, K, V) -> helion.Config
# block_sizes=[block_v] since only hl.register_block_size(dstate) is tunable
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 2, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    (2, 128, 4, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    (1, 256, 4, 64, 128): helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    # Benchmark shapes
    (1, 64, 1, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    (2, 512, 3, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    (2, 1024, 3, 64, 64): helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
}
# EVOLVE-BLOCK-END


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    q, k, v_new, h, g = data
    B, T, H, K = q.shape
    V = v_new.shape[-1]
    scale = K ** -0.5
    C = CHUNK_SIZE
    NT = T // C

    # Reshape to chunked form: [B, NT, C, H, ...]
    q_c = q.reshape(B, NT, C, H, K).contiguous()
    k_c = k.reshape(B, NT, C, H, K).contiguous()
    v_c = v_new.reshape(B, NT, C, H, V).contiguous()
    g_c = g.reshape(B, NT, C, H).contiguous()

    kernel = _KERNELS[(B, T, H, K, V)]
    out_c = kernel(q_c, k_c, v_c, h, g_c, scale, C)
    return out_c.reshape(B, T, H, V)
