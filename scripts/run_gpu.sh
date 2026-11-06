#!/usr/bin/env bash
set -euo pipefail

# Run inside the reviewed VM after installation; creates no cloud resources.
PYTHON=${PRISM_PYTHON:-.venv/bin/python}
RUN_DIR=${1:?Usage: bash scripts/run_gpu.sh runs/gpu-unique}
test ! -e "$RUN_DIR" || { echo "Choose a fresh output directory" >&2; exit 1; }
"$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "Real CUDA required; no fallback"'
mkdir -p "$RUN_DIR"
nvidia-smi > "$RUN_DIR/nvidia-smi.txt"
"$PYTHON" -m pip freeze > "$RUN_DIR/requirements-executed.txt"
"$PYTHON" -m pytest -q -m 'not gpu' -W error | tee "$RUN_DIR/cpu-contracts.txt"
"$PYTHON" -m pytest -q -m gpu -W error | tee "$RUN_DIR/gpu-tests.txt"
"$PYTHON" scripts/benchmark_gpu.py --output "$RUN_DIR/kernel"
for backend in native compiled triton; do
  "$PYTHON" scripts/http_experiment.py --device cuda --preprocess "$backend" \
    --requests 100 --batches 1 8 32 --output "$RUN_DIR/http-$backend.json"
done
"$PYTHON" -m prismserve extract --device cuda --output "$RUN_DIR/features.npz"
"$PYTHON" -m prismserve train --features "$RUN_DIR/features.npz" --output "$RUN_DIR/trained-head"
"$PYTHON" -m prismserve evaluate --device cuda --features "$RUN_DIR/features.npz" \
  --artifact "$RUN_DIR/trained-head" --output "$RUN_DIR/vision-evaluation.json"
"$PYTHON" scripts/pack_evidence.py "$RUN_DIR" gpu-evidence.tar.gz
echo "GPU gates executed. Review measured performance and any regressions before claiming speedup."
