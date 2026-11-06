"""Frozen-backbone transfer learning with explicit selection and assessment splits."""

import io
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from prismserve.data import IMAGENET_IDS, LABELS, WEIGHTS_SHA, digest, read_manifest
from prismserve.model import VisionModel
from prismserve.preprocess import decode_image


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def environment(device):
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }


def extract(root, output, device="cpu"):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = read_manifest(root)
    model = VisionModel(root / "resnet18-f37072fd.pth", device=device)
    features, logits = [], []
    started = time.perf_counter()
    for start in range(0, len(manifest["records"]), 16):
        images = [
            decode_image((root / row["path"]).read_bytes())
            for row in manifest["records"][start : start + 16]
        ]
        values, scores = model.tensors(images)
        features.append(values.cpu().numpy())
        logits.append(scores.cpu().numpy())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            features=np.concatenate(features),
            logits=np.concatenate(logits),
            labels=np.array([r["class_id"] for r in manifest["records"]]),
            splits=np.array([r["split"] for r in manifest["records"]]),
            manifest_sha256=np.array(digest(root / "subset.json")),
            weights_sha256=np.array(WEIGHTS_SHA),
        )
    write_json(
        output.with_suffix(".json"),
        {
            "environment": environment(device),
            "rows": len(manifest["records"]),
            "seconds": time.perf_counter() - started,
            "features_sha256": digest(output),
            "manifest_sha256": digest(root / "subset.json"),
            "role": "fixed feature extraction; labels do not enter backbone",
        },
    )


def train_head(features, labels, splits, epochs=120):
    torch.manual_seed(41)
    torch.set_num_threads(1)
    # Never index the assessment rows inside the learning/selection function.
    train = torch.tensor(features[splits == "train"], dtype=torch.float32)
    validation = torch.tensor(features[splits == "validation"], dtype=torch.float32)
    targets = torch.tensor(labels[splits == "train"], dtype=torch.long)
    val_targets = torch.tensor(labels[splits == "validation"], dtype=torch.long)
    mean, std = train.mean(0), train.std(0).clamp_min(0.01)
    x, xv = (train - mean) / std, (validation - mean) / std
    candidates = []
    for decay in (0.0, 0.001, 0.01):
        torch.manual_seed(41)
        head = torch.nn.Linear(512, 10)
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.01, weight_decay=decay)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(x), targets)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            val_logits = head(xv)
            accuracy = float((val_logits.argmax(1) == val_targets).float().mean())
            val_loss = float(torch.nn.functional.cross_entropy(val_logits, val_targets))
        candidates.append((accuracy, -val_loss, decay, head))
    accuracy, neg_loss, decay, head = max(candidates, key=lambda value: value[:3])
    with torch.no_grad():
        weight = head.weight / std
        bias = head.bias - weight @ mean
        error = float((validation @ weight.T + bias - head(xv)).abs().max())
    result = {"weight": weight.detach().numpy(), "bias": bias.detach().numpy()}
    report = {
        "seed": 41,
        "epochs": epochs,
        "optimizer": "AdamW",
        "learning_rate": 0.01,
        "selection": "max validation accuracy, then min cross entropy, then larger decay",
        "training_rows": len(train),
        "validation_rows": len(validation),
        "selected_weight_decay": decay,
        "folded_max_error": error,
        "validation_accuracy": accuracy,
        "validation_cross_entropy": -neg_loss,
        "candidates": [
            {"weight_decay": d, "accuracy": a, "cross_entropy": -loss}
            for a, loss, d, _ in candidates
        ],
    }
    return result, report


def train(features_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with np.load(features_path, allow_pickle=False) as data:
        head, report = train_head(data["features"], data["labels"], data["splits"])
        report["manifest_sha256"] = str(data["manifest_sha256"])
        report["weights_sha256"] = str(data["weights_sha256"])
    report["feature_file_sha256"] = digest(features_path)
    np.savez(output / "head.npz", **head)
    report["head_sha256"] = digest(output / "head.npz")
    report["assessment_used"] = False
    write_json(output / "training.json", report)


def metrics(logits, labels):
    probabilities = torch.tensor(logits).softmax(1).numpy()
    prediction = probabilities.argmax(1)
    correct = prediction == labels
    confidence = probabilities.max(1)
    confusion = np.zeros((10, 10), dtype=int)
    np.add.at(confusion, (labels, prediction), 1)
    bins = []
    for low in np.arange(0, 1, 0.1):
        mask = (confidence >= low) & (confidence < low + 0.1 if low < 0.9 else confidence <= 1)
        bins.append(
            {
                "lower": float(low),
                "count": int(mask.sum()),
                "accuracy": float(correct[mask].mean()) if mask.any() else None,
                "confidence": float(confidence[mask].mean()) if mask.any() else None,
            }
        )
    ece = sum(
        b["count"] / len(labels) * abs(b["accuracy"] - b["confidence"]) for b in bins if b["count"]
    )
    return {
        "n": len(labels),
        "accuracy": float(correct.mean()),
        "ece_10_bins": ece,
        "cross_entropy": float(
            -np.log(probabilities[np.arange(len(labels)), labels].clip(1e-12)).mean()
        ),
        "confusion": confusion.tolist(),
        "calibration_bins": bins,
        "per_class": [
            {
                "label": LABELS[i],
                "n": int((labels == i).sum()),
                "recall": float(correct[labels == i].mean()),
            }
            for i in range(10)
        ],
    }


def evaluate(root, features_path, artifact, output, device="cpu"):
    root, artifact = Path(root), Path(artifact)
    manifest = read_manifest(root)
    training = json.loads((artifact / "training.json").read_text())
    if digest(artifact / "head.npz") != training["head_sha256"]:
        raise ValueError("frozen head changed after selection")
    if digest(features_path) != training["feature_file_sha256"]:
        raise ValueError("feature cache changed after selection")
    with np.load(features_path, allow_pickle=False) as data:
        if str(data["manifest_sha256"]) != digest(root / "subset.json"):
            raise ValueError("features and manifest differ")
        mask = data["splits"] == "test"
        features, logits, labels = (
            data["features"][mask],
            data["logits"][mask],
            data["labels"][mask],
        )
    with np.load(artifact / "head.npz", allow_pickle=False) as head:
        learned = features @ head["weight"].T + head["bias"]
    mapped = logits[:, IMAGENET_IDS]
    stock_correct = np.argmax(mapped, axis=1) == labels
    learned_correct = learned.argmax(1) == labels
    rng = np.random.default_rng(417)
    deltas = learned_correct.astype(int) - stock_correct.astype(int)
    boot = [float(rng.choice(deltas, len(deltas), replace=True).mean()) for _ in range(2000)]
    records = [row for row in manifest["records"] if row["split"] == "test"]
    failures = []
    confidence = torch.tensor(learned).softmax(-1).max(-1).values.numpy()
    for i in np.argsort(-confidence):
        if not learned_correct[i]:
            failures.append(
                {
                    "path": records[i]["path"],
                    "expected": LABELS[labels[i]],
                    "predicted": LABELS[learned[i].argmax()],
                    "confidence": float(confidence[i]),
                    "mapped_baseline_correct": bool(stock_correct[i]),
                }
            )
    # Predeclared corruption slice: radius 2 Gaussian blur applied before shared geometry.
    model = VisionModel(root / "resnet18-f37072fd.pth", device=device, head=artifact / "head.npz")
    blurred_logits = []
    for start in range(0, len(records), 16):
        images = []
        for record in records[start : start + 16]:
            with Image.open(root / record["path"]) as original:
                image = original.convert("RGB").filter(ImageFilter.GaussianBlur(2))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                images.append(decode_image(buffer.getvalue()))
        _, scores = model.tensors(images)
        blurred_logits.append(scores.cpu().numpy())
    delta = float(deltas.mean())
    report = {
        "environment": environment(device),
        "manifest_sha256": digest(root / "subset.json"),
        "artifact_head_sha256": training["head_sha256"],
        "qa": manifest["qa"],
        "class_mapping": [
            {"label": label, "imagenet_id": i}
            for label, i in zip(LABELS, IMAGENET_IDS, strict=True)
        ],
        "unrestricted_imagenet_1000_accuracy": float(
            (logits.argmax(1) == np.array(IMAGENET_IDS)[labels]).mean()
        ),
        "mapped_pretrained_10_class": metrics(mapped, labels),
        "trained_head": metrics(learned, labels),
        "head_minus_mapped_accuracy": delta,
        "paired_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "head_demonstrates_improvement": bool(np.quantile(boot, 0.025) > 0),
        "blur_radius_2": metrics(np.concatenate(blurred_logits), labels),
        "failures_by_descending_confidence": failures,
        "limits": manifest["limits"]
        + [
            "Bootstrap assumes independent selected images",
            "No independent labels, unseen-class or open-set evaluation",
            "Softmax values are not calibrated deployment probabilities",
        ],
        "gpu_evidence": "pending"
        if device == "cpu"
        else "evaluation only; see separate serving/kernel gates",
    }
    write_json(output, report)
    return report
