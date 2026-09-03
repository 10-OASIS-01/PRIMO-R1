#!/usr/bin/env bash
# PRIMO-R1 environment setup.
#
# Assumes an activated Python 3.11 environment, e.g.
#   conda create -n primo-r1 python=3.11 && conda activate primo-r1
#
# Install order matters and is not arbitrary:
#   1. r1-v pulls in torch / vllm / trl and the eval dependencies.
#   2. The vendored qwen-vl-utils must be installed editable AFTER r1-v, so it
#      shadows any copy a transitive dependency dragged in from PyPI.
#   3. The vendored transformers-main tree must be installed LAST, so nothing
#      later replaces it. Qwen2.5-VL support shifts between upstream transformers
#      releases; a different version is the usual cause of shape/processor errors.
#
# Usage: bash setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

echo "[primo] 1/4 installing src/r1-v (editable, with dev extras)"
pip install -e "src/r1-v[dev]"

echo "[primo] 2/4 installing extras not declared as package dependencies"
# tensorboardX is optional logging; flash-attn needs an existing torch install,
# hence --no-build-isolation.
pip install tensorboardx
pip install flash-attn --no-build-isolation

echo "[primo] 3/4 installing vendored qwen-vl-utils (editable, with decord)"
# Editable and after r1-v, so this local copy wins over any PyPI qwen_vl_utils.
pip install -e "src/qwen-vl-utils[decord]"

echo "[primo] 4/4 installing vendored transformers-main"
# Required, and deliberately last. Do not replace with a PyPI transformers.
pip install ./transformers-main

echo "[primo] done. Verify with:"
echo "  python -c 'import transformers, trl, vllm, qwen_vl_utils; print(transformers.__version__, trl.__version__, vllm.__version__)'"
