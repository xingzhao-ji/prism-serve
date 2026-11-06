# NVIDIA execution guide — unverified

The accepted public release covers CPU behavior. The CUDA/Triton implementation and the following commands have not been validated on NVIDIA hardware. This guide assumes an existing Linux NVIDIA host supplied by its operator; it does not create cloud resources or prescribe an account, paid billing change or spending allowance.

Use a separate disposable checkout/output directory. Install Python 3.12, the pinned PyTorch 2.9.0 / torchvision 0.24.0 CUDA 12.8 wheel pair, common dependencies and this package:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements-common.lock
.venv/bin/python -m pip install --no-deps .
.venv/bin/python -m prismserve prepare
bash scripts/run_gpu.sh runs/gpu-001
```

The scripts require a real CUDA device, run CPU contracts plus 11 GPU tests, compare native/compiled/Triton preprocessing and tensor inference, save profiler traces and run loopback HTTP workloads. Geometry and labeled inputs stay comparable. The small linear-head optimization still uses CPU; this is not GPU head-training evidence. Installation, driver compatibility and all NVIDIA results remain unverified.

Stop on failed gates. Retain slowdowns, compilation costs, memory observations, failed requests and the strongest library baseline. A fast preprocessing kernel alone does not establish faster complete inference. The evidence packer includes selected JSON/text/NPZ artifacts with a 128 MiB expanded-content cap and excludes raw images and pretrained weights. Review the actual outputs before making GPU claims.

The original account-specific cloud proposal and two tests of that proposal are intentionally excluded from the public package. Historical CPU reports and their original source hashes remain historical; they do not identify this edited guide.
