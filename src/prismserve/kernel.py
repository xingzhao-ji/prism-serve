"""Fused uint8 NHWC -> FP32 normalized contiguous NCHW.

Each program owns a tile of output spatial positions for one (batch, channel)
plane. Input strides are runtime values, supporting transposed/cropped NHWC
views. Output writes are coalesced; RGB interleaved reads have stride three.
No resize/crop arithmetic is hidden in this kernel. GPU verification is required.
"""

import torch
import triton
import triton.language as tl

from prismserve.preprocess import validate_tensor


@triton.jit
def _normalize(
    X,
    Y,
    H: tl.constexpr,
    W: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    plane = tl.program_id(1)
    batch, channel = plane // 3, plane % 3
    source = batch * S0 + (position // W) * S1 + (position % W) * S2 + channel * S3
    value = tl.load(X + source, mask=position < H * W, other=0).to(tl.float32)
    mean = tl.where(channel == 0, 0.485, tl.where(channel == 1, 0.456, 0.406))
    std = tl.where(channel == 0, 0.229, tl.where(channel == 1, 0.224, 0.225))
    value = ((value / 255.0) - mean) / std
    tl.store(Y + plane * H * W + position, value, mask=position < H * W)


def normalize_triton(x: torch.Tensor) -> torch.Tensor:
    validate_tensor(x)
    if x.device.type != "cuda":
        raise ValueError("Triton path requires CUDA; no fallback")
    n, h, w, _ = x.shape
    output = torch.empty((n, 3, h, w), dtype=torch.float32, device=x.device)
    with torch.cuda.device(x.device):
        _normalize[(triton.cdiv(h * w, 256), n * 3)](
            x, output, h, w, *x.stride(), BLOCK=256, enable_fp_fusion=False
        )
    return output
