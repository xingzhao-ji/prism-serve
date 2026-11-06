import io
import zipfile

import numpy as np
import pytest
import torch

from prismserve.model import VisionModel


def test_model_rejects_untrusted_bytes_before_deserialization(tmp_path):
    weights = tmp_path / "untrusted.pth"
    weights.write_bytes(b"not the pinned pretrained state dictionary")
    with pytest.raises(ValueError, match="SHA-256"):
        VisionModel(weights)


def test_cpu_cannot_masquerade_as_gpu_backend(tmp_path):
    with pytest.raises(ValueError, match="require CUDA"):
        VisionModel(tmp_path / "missing", device="cpu", preprocess="triton")


def test_head_rejects_finite_parameters_that_overflow_float32():
    from prismserve.model import load_head_arrays

    buffer = io.BytesIO()
    np.savez(buffer, weight=np.full((10, 512), 1e100), bias=np.zeros(10))
    with pytest.raises(ValueError, match="float32"):
        load_head_arrays(buffer.getvalue())


def test_head_rejects_compressed_array_inflation_before_numpy_load():
    from prismserve.model import load_head_arrays

    buffer = io.BytesIO()
    np.savez_compressed(buffer, weight=np.zeros((1000, 512)), bias=np.zeros(10))
    with pytest.raises(ValueError, match="uncompressed"):
        load_head_arrays(buffer.getvalue())


def test_representable_head_matmul_overflow_fails_execution():
    class Features(torch.nn.Module):
        def forward(self, x):
            return torch.full((len(x), 512, 1, 1), 2.0)

    model = VisionModel.__new__(VisionModel)
    model.device = torch.device("cpu")
    model.normalize = lambda x: x
    model.features = Features()
    model.classifier = torch.nn.Linear(512, 10)
    with torch.no_grad():
        model.classifier.weight.fill_(3e38)
        model.classifier.bias.zero_()
    with pytest.raises(RuntimeError, match="nonfinite"):
        model.tensors([torch.zeros((224, 224, 3), dtype=torch.uint8)])


def test_head_rejects_forged_array_shape_before_allocating(monkeypatch):
    from prismserve.model import load_head_arrays

    member = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        member, {"descr": "<f4", "fortran_order": False, "shape": (1_000_000_000,)}
    )
    bias = io.BytesIO()
    np.save(bias, np.zeros(10, dtype=np.float32))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as file:
        file.writestr("weight.npy", member.getvalue())
        file.writestr("bias.npy", bias.getvalue())

    def allocation_guard(*args, **kwargs):
        raise RuntimeError("unsafe NumPy archive load attempted")

    monkeypatch.setattr(np, "load", allocation_guard)
    with pytest.raises(ValueError, match="shape"):
        load_head_arrays(archive.getvalue())


@pytest.mark.parametrize("fortran", [False, True])
def test_valid_head_storage_order_preserves_values(fortran):
    from prismserve.model import load_head_arrays

    expected = np.arange(5120, dtype=np.float64).reshape(10, 512) / 10000
    weight = np.asfortranarray(expected) if fortran else expected
    archive = io.BytesIO()
    np.savez(archive, weight=weight, bias=np.arange(10, dtype=np.float32))
    actual, bias = load_head_arrays(archive.getvalue())
    np.testing.assert_array_equal(actual, expected.astype(np.float32))
    np.testing.assert_array_equal(bias, np.arange(10, dtype=np.float32))
