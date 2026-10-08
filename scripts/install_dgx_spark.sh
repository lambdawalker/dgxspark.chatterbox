#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != Linux || "$(uname -m)" != aarch64 ]]; then
    echo "This installer targets DGX Spark (Linux aarch64)." >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is missing. Install/update the DGX Spark NVIDIA driver first." >&2
    exit 1
fi
nvidia-smi

# uv installs Python when needed and keeps the environment inside this checkout.
uv sync --locked --python 3.12
uv pip check --python .venv/bin/python
uv run --no-sync python scripts/check_dgx_spark.py

echo "Setup checks passed. Generate a test WAV with:"
echo "  uv run --no-sync python scripts/check_dgx_spark.py --synthesize"
