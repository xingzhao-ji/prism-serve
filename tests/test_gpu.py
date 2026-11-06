import pytest
import torch

from prismserve.preprocess import normalize_native

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def require_gpu():
    assert torch.cuda.is_available(), "GPU gate requires a real CUDA device; CPU/MPS cannot pass"


@pytest.mark.parametrize(
    "shape", [(1, 1, 1, 3), (1, 17, 33, 3), (8, 224, 224, 3), (32, 224, 224, 3)]
)
@pytest.mark.parametrize("strided", [False, True])
def test_triton_layout_normalization_parity(shape, strided):
    from prismserve.kernel import normalize_triton

    x = torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")
    if strided:
        x = x.transpose(1, 2)
    output = normalize_triton(x)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, normalize_native(x), atol=5e-7, rtol=1e-6)
    assert output.is_contiguous()


def test_triton_rejects_cpu_instead_of_falling_back():
    from prismserve.kernel import normalize_triton

    with pytest.raises(ValueError, match="CUDA"):
        normalize_triton(torch.zeros((1, 2, 2, 3), dtype=torch.uint8))


def test_triton_respects_nondefault_cuda_stream():
    from prismserve.kernel import normalize_triton

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x = torch.randint(0, 256, (1, 17, 31, 3), device="cuda", dtype=torch.uint8)
        actual = normalize_triton(x)
        expected = normalize_native(x)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=5e-7, rtol=1e-6)


def test_real_image_logits_and_class_mapping_match_native():
    import json
    from pathlib import Path

    from prismserve.model import VisionModel
    from prismserve.preprocess import decode_image

    root = Path("data")
    manifest = json.loads((root / "subset.json").read_text())
    records = [row for row in manifest["records"] if row["split"] == "test"][:8]
    images = [decode_image((root / row["path"]).read_bytes()) for row in records]
    native = VisionModel(root / "resnet18-f37072fd.pth", device="cuda")
    triton = VisionModel(root / "resnet18-f37072fd.pth", device="cuda", preprocess="triton")
    _, reference = native.tensors(images)
    _, actual = triton.tensors(images)
    torch.testing.assert_close(actual, reference, atol=1e-4, rtol=1e-4)
    assert torch.equal(actual.argmax(1), reference.argmax(1))
    assert native.labels == triton.labels
