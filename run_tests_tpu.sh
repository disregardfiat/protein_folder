#!/bin/bash
# Run validation and hierarchical tests; use TPU if JAX sees it.
#
# If you have a TPU dongle / local TPU, set before running (examples for Cloud TPU):
#   export TPU_ACCELERATOR_TYPE=v5e
#   export TPU_WORKER_HOSTNAMES=10.0.0.2
# For other setups, see: https://cloud.google.com/tpu/docs/jax-pods
#
# JAX_ENABLE_X64=1 enables float64 in JAX (recommended for this codebase).

set -e
cd "$(dirname "$0")"
ROOT="$(pwd)"
VENV="${ROOT}/.venv"
PY="${VENV}/bin/python"

if [[ ! -x "$PY" ]]; then
  echo "No .venv found. Create with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-tpu.txt"
  exit 1
fi

export JAX_ENABLE_X64=1
# Optional: force CPU if TPU init is noisy and you only want to test logic
# export JAX_PLATFORMS=cpu

echo "=== Running validation (TPU used if available) ==="
"$PY" -m horizon_physics.proteins.validation 2>/dev/null || "$PY" -m horizon_physics.proteins.validation

echo ""
echo "=== Checking JAX devices ==="
"$PY" -c "
import jax
devs = jax.devices()
tpu = [d for d in devs if d.platform == 'tpu']
print('Devices:', [str(d) for d in devs])
print('TPU count:', len(tpu))
" 2>/dev/null || true
