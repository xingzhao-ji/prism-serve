"""Pinned weights only; no request-supplied artifacts and no implicit network loading."""

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import torch
from torchvision.models import ResNet18_Weights, resnet18

from prismserve.data import LABELS, WEIGHTS_SHA
from prismserve.preprocess import normalize_native


def load_head_arrays(payload):
    """Bound expanded NPZ bytes and validate the precision actually executed."""
    if len(payload) > 1024 * 1024:
        raise ValueError("head exceeds artifact cap")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if {m.filename for m in members} != {"weight.npy", "bias.npy"} or len(members) != 2:
                raise ValueError("head file set mismatch")
            if sum(m.file_size for m in members) > 128 * 1024:
                raise ValueError("head uncompressed size exceeds cap")
            arrays = []
            for name, expected_shape in (("weight.npy", (10, 512)), ("bias.npy", (10,))):
                with archive.open(name) as member:
                    data = member.read(128 * 1024 + 1)
                if len(data) > 128 * 1024:
                    raise ValueError("head uncompressed size exceeds cap")
                stream = io.BytesIO(data)
                version = np.lib.format.read_magic(stream)
                readers = {
                    (1, 0): np.lib.format.read_array_header_1_0,
                    (2, 0): np.lib.format.read_array_header_2_0,
                }
                if version not in readers:
                    raise ValueError("unsupported head NPY version")
                shape, fortran, dtype = readers[version](stream, max_header_size=4096)
                if shape != expected_shape:
                    raise ValueError("head shape mismatch")
                if dtype not in (np.dtype("float32"), np.dtype("float64")):
                    raise ValueError("head must use float32 or float64 arrays")
                body = data[stream.tell() :]
                if len(body) != int(np.prod(expected_shape)) * dtype.itemsize:
                    raise ValueError("head array payload size mismatch")
                # Construct only after validating shape, dtype and actual bytes.
                # np.load would allocate from an untrusted NPY shape first.
                value = np.frombuffer(body, dtype=dtype).reshape(
                    expected_shape, order="F" if fortran else "C"
                )
                if not np.isfinite(value).all() or np.any(np.abs(value) > np.finfo(np.float32).max):
                    raise ValueError("head must contain finite float32-representable values")
                arrays.append(value.astype(np.float32))
            return tuple(arrays)
    except (zipfile.BadZipFile, EOFError) as exc:
        raise ValueError("invalid head archive") from exc


class VisionModel:
    def __init__(self, weights, *, device="cpu", preprocess="native", head=None):
        if device not in {"cpu", "cuda"} or preprocess not in {"native", "triton", "compiled"}:
            raise ValueError("unsupported device or preprocessing backend")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU/MPS fallback")
        if preprocess != "native" and device != "cuda":
            raise ValueError("compiled/Triton benchmark paths require CUDA")
        path = Path(weights)
        if not path.is_file() or path.stat().st_size > 50 * 1024 * 1024:
            raise ValueError("invalid weights file")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != WEIGHTS_SHA:
            raise ValueError("pretrained artifact SHA-256 mismatch")
        torch.set_num_threads(4)
        self.device = torch.device(device)
        network = resnet18(weights=None)
        network.load_state_dict(
            torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
        )
        self.features = torch.nn.Sequential(*list(network.children())[:-1]).eval().to(self.device)
        self.classifier = network.fc.eval().to(self.device)
        self.labels = list(ResNet18_Weights.IMAGENET1K_V1.meta["categories"])
        model_id = WEIGHTS_SHA
        if head:
            head_path = Path(head)
            if head_path.stat().st_size > 1024 * 1024:
                raise ValueError("head exceeds artifact cap")
            head_bytes = head_path.read_bytes()
            weight, bias = load_head_arrays(head_bytes)
            self.classifier = torch.nn.Linear(512, 10).eval().to(self.device)
            self.classifier.load_state_dict(
                {"weight": torch.from_numpy(weight), "bias": torch.from_numpy(bias)}
            )
            self.labels = list(LABELS)
            model_id = hashlib.sha256(payload + head_bytes).hexdigest()
        self.normalize = normalize_native
        if preprocess == "triton":
            from prismserve.kernel import normalize_triton

            self.normalize = normalize_triton
        elif preprocess == "compiled":
            self.normalize = torch.compile(normalize_native, fullgraph=True)
        self.metadata = {
            "model_id": model_id,
            "weights_sha256": WEIGHTS_SHA,
            "architecture": "ResNet18",
            "head": "trained-10" if head else "ImageNet-1000",
            "device": device,
            "preprocessing": preprocess,
            "geometry": "PIL RGB; bilinear shorter side 256; center crop 224",
            "labels": self.labels,
            "torch": torch.__version__,
        }
        self.predict([torch.zeros((224, 224, 3), dtype=torch.uint8)])

    @torch.inference_mode()
    def tensors(self, images):
        pixels = torch.stack(images).to(self.device)
        features = self.features(self.normalize(pixels)).flatten(1)
        logits = self.classifier(features)
        if not torch.isfinite(features).all() or not torch.isfinite(logits).all():
            raise RuntimeError("nonfinite model features or logits")
        return features, logits

    @torch.inference_mode()
    def predict(self, images):
        features, logits = self.tensors(images)
        probs, indices = logits.softmax(-1).topk(5)
        if not torch.isfinite(torch.linalg.vector_norm(features, dim=1)).all():
            raise RuntimeError("nonfinite embedding norm")
        normalized = torch.nn.functional.normalize(features, dim=1)
        if not torch.isfinite(probs).all() or not torch.isfinite(normalized).all():
            raise RuntimeError("nonfinite model probabilities or embeddings")
        probs, indices, normalized = probs.cpu(), indices.cpu(), normalized.cpu()
        return [
            {
                "top5": [
                    {"class_id": int(i), "label": self.labels[i], "probability": float(p)}
                    for p, i in zip(ps, ids, strict=True)
                ],
                "embedding": vector.tolist(),
            }
            for ps, ids, vector in zip(probs, indices, normalized, strict=True)
        ]
