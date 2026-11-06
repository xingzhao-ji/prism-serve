"""Pinned public inputs and a deterministic, class-stratified research subset."""

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

DATA_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
DATA_SHA = "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5"
WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHTS_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
SYNSETS = (
    "n01440764",
    "n02102040",
    "n02979186",
    "n03000684",
    "n03028079",
    "n03394916",
    "n03417042",
    "n03425413",
    "n03445777",
    "n03888257",
)
IMAGENET_IDS = (0, 217, 482, 491, 497, 566, 569, 571, 574, 701)
LABELS = (
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, destination, expected, cap):
    destination = Path(destination)
    if destination.exists():
        if digest(destination) != expected:
            raise ValueError(f"hash mismatch: {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("xb") as target:
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > cap:
                    raise ValueError("download exceeded byte cap")
                target.write(block)
        if digest(temporary) != expected:
            raise ValueError("download SHA-256 mismatch")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def safe_extract(source, destination):
    destination = Path(destination)
    with tarfile.open(source, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 30000 or sum(m.size for m in members) > 512 * 1024 * 1024:
            raise ValueError("archive exceeds extraction limits")
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError("unsafe archive entry")
        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination, members=members, filter="data")


def prepare(root):
    root = Path(root)
    download(DATA_URL, root / "imagenette2-160.tgz", DATA_SHA, 110 * 1024 * 1024)
    download(WEIGHTS_URL, root / "resnet18-f37072fd.pth", WEIGHTS_SHA, 50 * 1024 * 1024)
    if not (root / "imagenette2-160").exists():
        safe_extract(root / "imagenette2-160.tgz", root)
    return make_manifest(root)


def make_manifest(root):
    root = Path(root)
    records, seen_bytes, seen_pixels, duplicates = [], {}, {}, []
    # Stable path hash chooses 80 training-source images/class and 30 held-out
    # official-validation images/class. No predictions affect selection.
    for class_id, synset in enumerate(SYNSETS):
        for source, count in (("train", 80), ("val", 30)):
            candidates = list((root / "imagenette2-160" / source / synset).glob("*.JPEG"))
            candidates.sort(key=lambda p: hashlib.sha256(p.name.encode()).hexdigest())
            if len(candidates) < count:
                raise ValueError("dataset incomplete")
            for index, path in enumerate(candidates[:count]):
                split = "test" if source == "val" else ("train" if index < 60 else "validation")
                raw_sha = digest(path)
                with Image.open(path) as image:
                    width, height = image.size
                    pixels = np.array(image.convert("RGB"))
                pixel_sha = hashlib.sha256(
                    str(pixels.shape).encode() + pixels.tobytes()
                ).hexdigest()
                relative = str(path.relative_to(root))
                for key, seen in ((raw_sha, seen_bytes), (pixel_sha, seen_pixels)):
                    if key in seen:
                        duplicates.append([seen[key], relative])
                    seen[key] = relative
                records.append(
                    {
                        "path": relative,
                        "sha256": raw_sha,
                        "pixel_sha256": pixel_sha,
                        "width": width,
                        "height": height,
                        "class_id": class_id,
                        "synset": synset,
                        "split": split,
                    }
                )
    if duplicates:
        raise ValueError(f"selected subset contains exact duplicates: {duplicates[:5]}")
    manifest = {
        "schema": 1,
        "archive_sha256": DATA_SHA,
        "records": records,
        "qa": {
            "selected_count": len(records),
            "exact_byte_duplicates": 0,
            "exact_pixel_duplicates": 0,
            "near_duplicate_review": "not performed; perceptual leakage remains possible",
            "label_source": "original folder synset; not human relabeled",
        },
        "limits": [
            "ImageNet pretraining overlaps source training images",
            "Official val is held out from this head, not unseen in model development",
            "Small balanced subset is not a representative deployment sample",
        ],
    }
    target = root / "subset.json"
    payload = json.dumps(manifest, indent=2) + "\n"
    if target.exists() and target.read_text() != payload:
        raise ValueError("existing subset manifest differs")
    target.write_text(payload)
    return manifest


def read_manifest(root):
    root = Path(root)
    manifest = json.loads((root / "subset.json").read_text())
    for record in manifest["records"]:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != record["sha256"]:
            raise ValueError("dataset manifest integrity failure")
    return manifest
