# Verification record

Recorded locally on 2026-10-23, Python 3.12.12, PyTorch 2.9.0, torchvision 0.24.0, macOS 27.0.1 arm64. Device explicitly `cpu`; CUDA unavailable. The developer machine is an Apple M4 Pro. Full installed package versions are in `requirements-macos.lock`.

| Check | Actual result |
|---|---|
| Ruff format and lint, source/scripts/tests | Passed |
| CPU suite, `pytest -q -m 'not gpu' -W error` | **40 passed, 11 GPU tests deselected** |
| Exact official pretrained preprocessing / independent NumPy reference | Passed CPU tests |
| Scheduler admission, ordering, cancellation, expiry, worker failure, shutdown | Passed CPU tests |
| Timed-out decoder retains capacity until underlying thread completes | Passed concurrency regression test |
| Malformed weights rejected before deserialization | Passed |
| Test features/labels cannot influence head training or selection | Passed mutation test |
| Dataset selection and byte/pixel duplicate QA | 1,100 images, no exact selected duplicates |
| Real pretrained model feature extraction | 1,100 images, 116.7 seconds on CPU |
| Real transfer-head training and held-out labeled evaluation | Completed; improvement gate failed honestly |
| Real subprocess HTTP service | 360/360 normal requests succeeded in final matrix; all four overload drills recovered |
| GPU benchmark invoked without CUDA | Exited 1 with explicit failure; no GPU output directory created |
| CUDA / Triton / profiler / GPU HTTP | **Pending actual NVIDIA execution** |
| Hosted CI | Defined, not run on a hosted runner |
| Independent review | CPU scope accepted after exact training/assessment replay and three finding rechecks; GPU pending |

Tests were written before their corresponding initial implementation groups. Initial runs reported missing modules during collection; after implementation one scheduler test caught absent zero-valued metric keys, which were fixed. This is evidence of contract development, not a claim that every test had a clean assertion-only red phase. The decoder cancellation case was added as a regression check while hardening bounded admission.

The first HTTP report is `evidence/cpu-http.json`. A second run, `cpu-http-recovery.json`, adds an explicit valid-request/readiness check after the overload queue drains; do not merge their timing samples or choose only favorable values. Both are shared-laptop observations with 30 requests per ordinary workload, five initial warmups, and separate 256-request 128-client/10 ms deadline drills.

GPU tests intentionally do not skip when selected without CUDA. Use `make test-cpu` for CPU contracts and `make test-gpu` only on a supported NVIDIA host with prepared data. A plain `pytest` on this laptop will fail the GPU gate; that is intentional and should not be represented as a fully green GPU suite.

Evaluation outputs carry input, weight, feature-file and head hashes. The retained head and training report are small research artifacts; raw images, pretrained weights, environments and feature caches are ignored. Changes to kernels, preprocessing, model execution or dependency versions invalidate relevant GPU/HTTP evidence and require a new output directory.

The final post-review matrix is `evidence/cpu-http-final.json`: 360 normal requests succeeded, all four overload drills recovered with prediction/readiness 200, and all 16 recorded scheduler snapshots satisfy the unique terminal-outcome invariant with no retained/unresolved work. Model-work lifecycle counters measure the first terminal resolution, not guaranteed HTTP response delivery. Earlier reports retain their original outcomes and timings; the intermediate `cpu-http-reviewed.json` contains the deadline double-counting bug and is superseded for that metric. Final single-client medians span 12.49–37.52 ms and concurrent throughput 27.88–96.53 successful requests/s; variation across short shared-host runs rules out a stable capacity claim.

Independent review closed resize amplification, precision/matmul overflow and duplicate terminal-outcome accounting. Added regressions were observed failing before their corresponding fixes. See `docs/independent-employer-review.md` and `evidence/independent-review.json`.

Final head-artifact review additionally found that a small NPY member could claim a huge array shape before NumPy validates its data length. The loader now bounds member/header bytes and validates shape, dtype and exact payload length before constructing an array. An independent no-allocation probe and eight valid dtype/order/version combinations passed; the retained trained head values remain identical. This head-loader-only change does not alter the pretrained no-head HTTP path measured in `cpu-http-final.json`.
