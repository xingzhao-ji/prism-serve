"""Real TCP/HTTP load matrix, admission errors and malformed-image recovery."""

import argparse
import asyncio
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import torch

from prismserve.experiment import environment, write_json


async def workload(base, image, concurrency, count, timeout_ms=30000):
    durations, statuses = [], []
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=base, timeout=40) as client:

        async def one():
            async with semaphore:
                start = time.perf_counter()
                response = await client.post(
                    f"/v1/predict?timeout_ms={timeout_ms}",
                    content=image,
                    headers={"content-type": "image/jpeg"},
                )
                durations.append((time.perf_counter() - start) * 1000)
                statuses.append(response.status_code)

        started = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(count)))
        elapsed = time.perf_counter() - started
        stats = (await client.get("/v1/stats")).json()
    return {
        "concurrency": concurrency,
        "requests": count,
        "p50_ms": float(np.median(durations)),
        "p95_ms": float(np.quantile(durations, 0.95)),
        "p99_ms": float(np.quantile(durations, 0.99)),
        "successful_requests_per_second": statuses.count(200) / elapsed,
        "status_counts": {str(status): statuses.count(status) for status in set(statuses)},
        "elapsed_seconds": elapsed,
        "server_cumulative_stats": stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--preprocess", choices=["native", "compiled", "triton"], default="native")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("FAIL: real CUDA device required; no fallback")
    if args.output.exists():
        raise SystemExit("output exists; choose a fresh evidence path")
    if not 10 <= args.requests <= 10000:
        raise SystemExit("requests must be 10..10000")
    manifest = json.loads((args.data / "subset.json").read_text())
    image = (
        args.data / next(r["path"] for r in manifest["records"] if r["split"] == "test")
    ).read_bytes()
    rows, failures = [], []
    for batch in args.batches:
        for wait in (0, 5):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            command = [
                sys.executable,
                "-m",
                "prismserve",
                "serve",
                "--weights",
                str(args.data / "resnet18-f37072fd.pth"),
                "--device",
                args.device,
                "--preprocess",
                args.preprocess,
                "--max-batch",
                str(batch),
                "--max-pending",
                "64",
                "--wait-ms",
                str(wait),
                "--port",
                str(port),
            ]
            process = subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            try:
                deadline = time.monotonic() + 180
                with httpx.Client(base_url=base, timeout=30) as client:
                    while True:
                        if process.poll() is not None:
                            raise RuntimeError("server exited before readiness")
                        try:
                            if client.get("/ready").status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        if time.monotonic() > deadline:
                            raise RuntimeError("model never became ready")
                        time.sleep(0.1)
                    metadata = client.get("/v1/model").json()
                    for _ in range(5):
                        response = client.post(
                            "/v1/predict?timeout_ms=30000",
                            content=image,
                            headers={"content-type": "image/jpeg"},
                        )
                        response.raise_for_status()
                    invalid = client.post(
                        "/v1/predict", content=b"bad", headers={"content-type": "image/jpeg"}
                    )
                    recovery = client.post(
                        "/v1/predict?timeout_ms=30000",
                        content=image,
                        headers={"content-type": "image/jpeg"},
                    )
                    assert invalid.status_code == 422 and recovery.status_code == 200
                    failures.append(
                        {
                            "batch": batch,
                            "wait_ms": wait,
                            "malformed": invalid.status_code,
                            "recovery": recovery.status_code,
                        }
                    )
                for concurrency in (1, 8, 32):
                    result = asyncio.run(workload(base, image, concurrency, args.requests))
                    result.update({"max_batch": batch, "wait_ms": wait})
                    rows.append(result)
                overload = asyncio.run(workload(base, image, 128, 256, timeout_ms=10))
                with httpx.Client(base_url=base, timeout=40) as client:
                    recovery_deadline = time.monotonic() + 30
                    while client.get("/v1/stats").json().get("pending", 0):
                        if time.monotonic() >= recovery_deadline:
                            raise RuntimeError("overload work failed to drain")
                        time.sleep(0.05)
                    recovered = client.post(
                        "/v1/predict?timeout_ms=30000",
                        content=image,
                        headers={"content-type": "image/jpeg"},
                    )
                    recovered.raise_for_status()
                    overload["subsequent_request_status"] = recovered.status_code
                    overload["readiness_after_recovery"] = client.get("/ready").status_code
                failures.append(
                    {"batch": batch, "wait_ms": wait, "overload_and_deadline": overload}
                )
            finally:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    write_json(
        args.output,
        {
            "environment": environment(args.device),
            "model": metadata,
            "rows": rows,
            "failure_drills": failures,
            "measurement": "real loopback HTTP; 5 warmups; client concurrency closed-loop",
            "limits": [
                "Single process, loopback network",
                "One fixed image reused for load",
                "HTTP timing includes decode/geometry, copy, queue, model, serialization",
            ],
        },
    )


if __name__ == "__main__":
    main()
