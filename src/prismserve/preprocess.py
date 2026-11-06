"""Decode/geometry are shared; only normalization/layout are replaced by Triton."""

import io
import warnings

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_PIXELS = 4_000_000
MAX_RESIZED_PIXELS = 1_048_576
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def decode_image(data: bytes) -> torch.Tensor:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("image must contain 1..2097152 bytes")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG"} or image.width * image.height > MAX_PIXELS:
                    raise ValueError("only JPEG/PNG up to 4 million pixels are accepted")
                if getattr(image, "n_frames", 1) != 1:
                    raise ValueError("animated images are not supported")
                # The shorter-side transform can amplify a tiny, very narrow
                # source into billions of pixels. Bound its projected output
                # before RGB conversion or resize allocates an intermediate.
                short, long = sorted((image.width, image.height))
                if short < 1 or 256 * (256 * long // short) > MAX_RESIZED_PIXELS:
                    raise ValueError("resize would exceed 1048576 intermediate pixels")
                rgb = image.convert("RGB")
                resized = F.resize(
                    rgb, [256], interpolation=InterpolationMode.BILINEAR, antialias=True
                )
                cropped = F.center_crop(resized, [224, 224])
                return torch.from_numpy(np.array(cropped, dtype=np.uint8, copy=True))
    except (
        OSError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("invalid image") from exc


def validate_tensor(x: torch.Tensor):
    if x.dtype != torch.uint8 or x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError("input must be uint8 NHWC with three RGB channels")
    if not 1 <= x.shape[0] <= 64 or not 1 <= x.shape[1] <= 4096 or not 1 <= x.shape[2] <= 4096:
        raise ValueError("input dimensions exceed contract")
    if x.numel() > 64 * 224 * 224 * 3:
        raise ValueError("input element limit exceeded")


def normalize_native(x: torch.Tensor) -> torch.Tensor:
    validate_tensor(x)
    # Same operation order as torchvision's official FP32 weights transform.
    value = x.permute(0, 3, 1, 2).to(torch.float32).contiguous().div(255.0)
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (value - mean) / std
