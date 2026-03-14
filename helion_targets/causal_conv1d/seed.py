from task import input_t, output_t

import torch
import torch.nn.functional as F
import helion
import helion.language as hl


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def causal_conv1d_kernel(
        x_pad: torch.Tensor,    # [N, S+W-1] flattened (B*D), zero-padded
        weight: torch.Tensor,   # [N, W] broadcast weights
        bias: torch.Tensor,     # [N] broadcast bias
        S_out: int,
    ) -> torch.Tensor:
        N, S_pad = x_pad.shape
        W = hl.specialize(weight.shape[1])
        S = hl.specialize(S_out)

        out = torch.empty(N, S, dtype=x_pad.dtype, device=x_pad.device)

        # EVOLVE-BLOCK-START
        for tile_n in hl.tile(N):
            for tile_s in hl.tile(S):
                acc = hl.zeros([tile_n, tile_s], dtype=torch.float32)
                for k in range(W):
                    w_k = weight[tile_n, k].to(torch.float32)
                    x_k = x_pad[tile_n, tile_s + k].to(torch.float32)
                    acc = acc + w_k[:, None] * x_k
                out[tile_n, tile_s] = acc + bias[tile_n].to(torch.float32)[:, None]
        # EVOLVE-BLOCK-END

        return out

    return causal_conv1d_kernel


# EVOLVE-BLOCK-START
# Per-shape configs: (B, D, S, W) -> helion.Config
# block_sizes has 2 entries: [tile_n_block, tile_s_block]
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 64, 64, 4): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=1),
    (2, 128, 128, 4): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=1),
    (1, 256, 256, 3): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=1),
    (1, 128, 64, 8): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=1),
    (4, 64, 128, 4): helion.Config(block_sizes=[64, 64], num_warps=4, num_stages=1),
    # Benchmark shapes
    (1, 1536, 2048, 4): helion.Config(block_sizes=[64, 128], num_warps=4, num_stages=2),
    (1, 2560, 2048, 4): helion.Config(block_sizes=[64, 128], num_warps=4, num_stages=2),
    (1, 2560, 4096, 4): helion.Config(block_sizes=[64, 128], num_warps=4, num_stages=2),
}
# EVOLVE-BLOCK-END


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]

    # Flatten B*D, pad, and broadcast weight/bias
    x_flat = x.reshape(B * D, S)
    x_pad = F.pad(x_flat, (W - 1, 0))  # [N, S+W-1]
    weight_broad = weight.unsqueeze(0).expand(B, D, W).reshape(B * D, W).contiguous()
    bias_broad = bias.unsqueeze(0).expand(B, D).reshape(B * D).contiguous()

    kernel = _KERNELS[(B, D, S, W)]
    result = kernel(x_pad, weight_broad, bias_broad, S)
    return result.reshape(B, D, S)
