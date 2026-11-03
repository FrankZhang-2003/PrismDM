#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH=".:${PYTHONPATH:-}"
# Start nodectl.py in a separate terminal before selecting a secure backend.
exec python scripts/flax_sample_hack.py "$@"
