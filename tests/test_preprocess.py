import io

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights

from prismserve.preprocess import decode_image, normalize_native


@pytest.mark.parametrize("shape", [(160, 211), (301, 160), (224, 224), (1, 16)])
def test_decode_normalize_matches_official_pretrained_transform(shape):
    rng = np.random.default_rng(42)
    image = Image.fromarray(rng.integers(0, 256, (*shape, 3), dtype=np.uint8))
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")
    result = normalize_native(decode_image(encoded.getvalue())[None])
    expected = ResNet18_Weights.IMAGENET1K_V1.transforms()(image)[None]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.is_contiguous()


def test_independent_numpy_normalization_and_channels():
    x = torch.tensor([[[[0, 127, 255], [13, 49, 200]]]], dtype=torch.uint8)
    expected = (
        (x.numpy().astype(np.float32) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    ).transpose(0, 3, 1, 2)
    np.testing.assert_allclose(normalize_native(x).numpy(), expected, rtol=1e-6)


@pytest.mark.parametrize(
    "value",
    [
        torch.zeros(2, 3, 4, 3),
        torch.zeros(0, 2, 2, 3, dtype=torch.uint8),
        torch.zeros(1, 3, 2, 4, dtype=torch.uint8),
    ],
)
def test_normalize_rejects_invalid_contract(value):
    with pytest.raises(ValueError):
        normalize_native(value)


def test_rejects_invalid_and_oversized_image_before_decode():
    with pytest.raises(ValueError):
        decode_image(b"not an image")
    with pytest.raises(ValueError):
        decode_image(b"x" * (2 * 1024 * 1024 + 1))


@pytest.mark.parametrize("size", [(1, 16384), (16384, 1), (1, 17)])
def test_rejects_resize_amplification_before_allocating(monkeypatch, size):
    from prismserve import preprocess

    image = Image.new("RGB", size, "red")
    encoded = io.BytesIO()
    image.save(encoded, format="PNG")

    def unsafe_resize(*args, **kwargs):
        raise RuntimeError("unbounded resize reached")

    monkeypatch.setattr(preprocess.F, "resize", unsafe_resize)
    with pytest.raises(ValueError, match="resize"):
        decode_image(encoded.getvalue())
