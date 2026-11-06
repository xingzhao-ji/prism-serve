"""GPU evidence gate. Never silently skips, uses CPU, or substitutes MPS."""

import argparse
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from prismserve.experiment import environment, write_json
from prismserve.model import VisionModel
from prismserve.preprocess import normalize_native


def sample(fn, repetitions=100):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repetitions)
    ]
    for start, end in events:
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    times = [start.elapsed_time(end) for start, end in events]
    return {
        "p50_ms": float(np.median(times)),
        "p95_ms": float(np.quantile(times, 0.95)),
        "samples_ms": times,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("data/resnet18-f37072fd.pth"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA GPU required. No CPU/MPS fallback; no GPU evidence produced.")
    if args.output.exists():
        raise SystemExit("output exists; preserve prior evidence and choose another run directory")
    import triton

    from prismserve.kernel import normalize_triton

    args.output.mkdir(parents=True)
    torch.manual_seed(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = VisionModel(args.weights, device="cuda")
    compiled = torch.compile(normalize_native, fullgraph=True)
    methods = {
        "torch_native": normalize_native,
        "torch_compile": compiled,
        "custom_triton": normalize_triton,
    }
    rows = []
    with torch.inference_mode():
        for batch in (1, 8, 32):
            x = torch.randint(0, 256, (batch, 224, 224, 3), device="cuda", dtype=torch.uint8)
            reference = normalize_native(x)
            reference_logits = model.classifier(model.features(reference).flatten(1))
            cold = {}
            for name, fn in methods.items():
                torch.cuda.synchronize()
                started = time.perf_counter()
                output = fn(x)
                torch.cuda.synchronize()
                cold[name] = (time.perf_counter() - started) * 1000
                torch.testing.assert_close(output, reference, atol=5e-7, rtol=1e-6)
                logits = model.classifier(model.features(output).flatten(1))
                torch.testing.assert_close(logits, reference_logits, atol=1e-4, rtol=1e-4)
                assert torch.equal(logits.argmax(1), reference_logits.argmax(1))
            for repeat in range(3):
                order = list(methods)
                random.Random(41 + repeat).shuffle(order)
                for name in order:
                    fn = methods[name]
                    kernel = sample(lambda fn=fn, x=x: fn(x))
                    pipeline = sample(
                        lambda fn=fn, x=x: model.classifier(model.features(fn(x)).flatten(1))
                    )
                    torch.cuda.reset_peak_memory_stats()
                    allocated_before = torch.cuda.memory_allocated()
                    fn(x)
                    torch.cuda.synchronize()
                    rows.append(
                        {
                            "batch": batch,
                            "repeat": repeat,
                            "method": name,
                            "normalization": kernel,
                            "tensor_pipeline": pipeline,
                            "tensor_pipeline_images_per_second": batch * 1000 / pipeline["p50_ms"],
                            "peak_extra_normalization_bytes": torch.cuda.max_memory_allocated()
                            - allocated_before,
                            "first_call_ms": cold[name],
                        }
                    )
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            for name, fn in methods.items():
                with torch.profiler.record_function(name):
                    for _ in range(5):
                        model.classifier(model.features(fn(x)).flatten(1))
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(args.output / "trace.json"))
    try:
        smi = subprocess.check_output(["nvidia-smi"], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        smi = f"unavailable: {exc}"
    report = {
        "environment": environment("cuda"),
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "triton": triton.__version__,
        "nvidia_smi": smi,
        "rows": rows,
        "parity": "passed",
        "timing": "CUDA events, 20 warmup +100 samples, 3 order-shuffled rounds",
        "excludes": ["PIL decode and geometry", "host-device transfer", "HTTP", "queue wait"],
        "performance_claim": "Compare with fastest library baseline; no assumed speedup",
        "trace": "trace.json",
    }
    write_json(args.output / "benchmark.json", report)
    print(f"Saved GPU measurements and trace under {args.output}")


if __name__ == "__main__":
    main()
