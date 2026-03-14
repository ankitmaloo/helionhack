from task import input_t, output_t

import torch
import helion
import helion.language as hl


FP8_MAX = 448.0
FP8_EPS = 1e-10


def _make_kernel(config: helion.Config):
    @helion.kernel(static_shapes=True, config=config)
    def fp8_quant_kernel(
        x: torch.Tensor,       # [N, group_size] input grouped rows
        x_s: torch.Tensor,     # [N] output scales
    ) -> torch.Tensor:
        N = x.size(0)
        G = hl.specialize(x.size(1))

        x_q = torch.empty_like(x)

        # EVOLVE-BLOCK-START
        for row_idx in hl.tile(N):
            row = x[row_idx, :].to(torch.float32)

            # Per-group absmax
            absmax = torch.amax(torch.abs(row), -1)
            absmax = torch.clamp(absmax, min=FP8_EPS)

            # Scale = absmax / fp8_max
            scale = absmax / FP8_MAX

            # Quantize: x / scale, clamped to FP8 range
            quantized = row / scale[:, None]
            quantized = torch.clamp(quantized, -FP8_MAX, FP8_MAX)

            x_q[row_idx, :] = quantized
            x_s[row_idx] = scale
        # EVOLVE-BLOCK-END

        return x_q

    return fp8_quant_kernel


# EVOLVE-BLOCK-START
# Per-shape configs: (num_tokens, hidden_dim, group_size) -> helion.Config
SHAPE_CONFIGS: dict[tuple, helion.Config] = {
    # Test shapes
    (1, 256, 64): helion.Config(block_sizes=[1], num_warps=4, num_stages=1),
    (4, 512, 128): helion.Config(block_sizes=[4], num_warps=4, num_stages=1),
    (16, 1024, 64): helion.Config(block_sizes=[16], num_warps=4, num_stages=1),
    (1, 4096, 128): helion.Config(block_sizes=[1], num_warps=4, num_stages=1),
    (8, 4096, 128): helion.Config(block_sizes=[8], num_warps=4, num_stages=1),
    # Benchmark shapes
    (256, 4096, 128): helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    (256, 8192, 128): helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
    (4096, 7168, 128): helion.Config(block_sizes=[64], num_warps=4, num_stages=1),
}
# EVOLVE-BLOCK-END


_KERNELS = {shape: _make_kernel(cfg) for shape, cfg in SHAPE_CONFIGS.items()}


def custom_kernel(data: input_t) -> output_t:
    x, x_q, x_s = data
    T, H = x.shape
    G = x_s.shape[1]
    gsz = H // G
    N = T * G

    kernel = _KERNELS[(T, H, gsz)]

    flat_in = x.reshape(N, gsz)
    flat_s = x_s.reshape(N)

    flat_q = kernel(flat_in, flat_s)

    x_q[...] = flat_q.reshape(T, H)
    x_s[...] = flat_s.reshape(T, G)
    return x_q, x_s
