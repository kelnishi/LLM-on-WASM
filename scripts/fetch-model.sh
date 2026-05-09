#!/usr/bin/env bash
# Download Gemma 3 270M ONNX + tokenizer from HuggingFace and merge the
# external-data weight blob into a single self-contained .onnx file. The
# wasi-nn `load` ABI takes raw bytes, so external_data references can't be
# resolved at runtime — we have to inline them ahead of time.
#
# Requires: curl, python3 with the `onnx` package (pip install onnx).
#
# Variant: VARIANT=q4 (default) | fp32 | fp16 | q4f16 | quantized

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models
cd models

REPO="onnx-community/gemma-3-270m-it-ONNX"
VARIANT="${VARIANT:-q4}"
OUT="gemma3_270m.onnx"

case "$VARIANT" in
    fp32)      BASE="model" ;;
    fp16)      BASE="model_fp16" ;;
    q4)        BASE="model_q4" ;;
    q4f16)     BASE="model_q4f16" ;;
    quantized) BASE="model_quantized" ;;
    *) echo "unknown VARIANT: $VARIANT (expected fp32|fp16|q4|q4f16|quantized)" >&2; exit 1 ;;
esac

URL="https://huggingface.co/${REPO}/resolve/main"

download() {
    local rel="$1"
    if [ -f "$rel" ]; then
        echo "✓ $rel already present"
    else
        echo "→ downloading $rel"
        mkdir -p "$(dirname "$rel")"
        curl -L --fail --progress-bar -o "$rel" "$URL/$rel"
    fi
}

download "tokenizer.json"
download "onnx/${BASE}.onnx"
download "onnx/${BASE}.onnx_data"

if [ ! -f "$OUT" ] || [ "onnx/${BASE}.onnx" -nt "$OUT" ] || [ "onnx/${BASE}.onnx_data" -nt "$OUT" ]; then
    echo "→ inlining external weights into $OUT (needs: pip install onnx)"
    python3 - "onnx/${BASE}.onnx" "$OUT" <<'PY'
import os, sys
import onnx
src, dst = sys.argv[1], sys.argv[2]
model = onnx.load(src, load_external_data=True)
onnx.save(model, dst, save_as_external_data=False)
print(f"✓ wrote {dst} ({os.path.getsize(dst)/1e6:.1f} MB)")
PY
else
    echo "✓ $OUT already up to date"
fi

echo
echo "done. ready to run:"
ls -lh tokenizer.json "$OUT"
