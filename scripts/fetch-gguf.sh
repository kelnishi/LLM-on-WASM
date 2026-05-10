#!/usr/bin/env bash
# Download a small GGUF model for the LlamaSharp wasi-nn harness.
# Default: Qwen2.5-0.5B-Instruct Q4_K_M (~352 MB) — small enough for
# CI / commodity laptops while still being a real chat model.
#
# Variant: VARIANT=q4 (default) | q5 | q8 | f16
#
# Files land in models/ under the filename the WACS LlamaSharp
# bindable expects (stem matches `WACS_WASINN_GGUF_DIR` scan).

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models
cd models

REPO="Qwen/Qwen2.5-0.5B-Instruct-GGUF"
VARIANT="${VARIANT:-q4}"

case "$VARIANT" in
    q4)  FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf" ;;
    q5)  FILE="qwen2.5-0.5b-instruct-q5_k_m.gguf" ;;
    q8)  FILE="qwen2.5-0.5b-instruct-q8_0.gguf"   ;;
    f16) FILE="qwen2.5-0.5b-instruct-fp16.gguf"   ;;
    *) echo "unknown VARIANT: $VARIANT (expected q4|q5|q8|f16)" >&2; exit 1 ;;
esac

URL="https://huggingface.co/${REPO}/resolve/main/${FILE}"

if [ -f "$FILE" ]; then
    echo "✓ $FILE already present"
else
    echo "→ downloading $FILE"
    curl -L --fail --progress-bar -o "$FILE" "$URL"
fi

# WACS LlamaSharp's filename-stem registry uses
# Path.GetFileNameWithoutExtension. The stem is what
# `load-by-name(...)` looks up.
STEM="${FILE%.gguf}"

echo
echo "done. ready to run with:"
echo "  export WACS_WASINN_GGUF_DIR=\$(pwd)"
echo "  wacs run target/wasm32-wasip2/release/wasi-nn-llm.wasm \\"
echo "      --wasip2 --bind Wacs.WASI.NN.LlamaSharp \\"
echo "      -d \$(pwd)::/models"
echo
echo "guest's load-by-name target:  $STEM"
ls -lh "$FILE"
