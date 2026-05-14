#!/usr/bin/env bash
# Download all-MiniLM-L6-v2 from HuggingFace and convert it to OpenVINO
# IR with a fixed [1, 64] input shape. wasi-nn requires concrete shapes
# (dynamic dimensions trip a RuntimeError at compile time), so we pin
# seq_len here at conversion time — long inputs get truncated guest-side.
#
# Requires: curl, python3.
#
# The OpenVINO native NuGets are at different versions per RID
# (macOS arm64 → 2026.1.0; Linux/Windows → 2024.4–2026.0). Newer
# OpenVINO runtimes read older IR fine, but the reverse isn't true,
# so we pin Python to the OLDEST native version we ship — that
# produces IR every platform's native runtime can load. The Linux
# pip wheels for 2024.4 are healthy; the macOS arm64 2024.4 wheel
# has a broken `__LINKEDIT`, so on macOS we pin to 2026.1 to match
# the local native there.

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models

case "$(uname -s)" in
    Darwin) DEFAULT_OV_VER="2026.1.0" ;;
    *)      DEFAULT_OV_VER="2024.4.0" ;;
esac
OPENVINO_PY_VERSION="${OPENVINO_PY_VERSION:-$DEFAULT_OV_VER}"

# Ensure a compatible openvino is importable. Reinstall pinned version
# only if the current install is absent or wrong.
HAVE_OV_VER="$(python3 -c 'import openvino,sys;print(openvino.__version__.split("-")[0])' 2>/dev/null || true)"
if [ "$HAVE_OV_VER" != "$OPENVINO_PY_VERSION" ]; then
    echo "→ installing openvino==$OPENVINO_PY_VERSION (current: ${HAVE_OV_VER:-none})"
    python3 -m pip install --quiet "openvino==$OPENVINO_PY_VERSION"
fi

REPO="sentence-transformers/all-MiniLM-L6-v2"
ONNX_SRC="models/minilm-onnx-fp32.onnx"
TOKENIZER="models/sentence-tokenizer.json"
IR_XML="models/minilm.xml"
IR_BIN="models/minilm.bin"
SEQ_LEN=64

URL_BASE="https://huggingface.co/${REPO}/resolve/main"

download() {
    local dst="$1" src="$2"
    if [ -f "$dst" ]; then
        echo "✓ $dst already present"
    else
        echo "→ downloading $src"
        curl -L --fail --progress-bar -o "$dst" "$URL_BASE/$src"
    fi
}

download "$ONNX_SRC" "onnx/model.onnx"
download "$TOKENIZER" "tokenizer.json"

if [ ! -f "$IR_XML" ] || [ ! -f "$IR_BIN" ] \
    || [ "$ONNX_SRC" -nt "$IR_XML" ]; then
    echo "→ converting ONNX → OpenVINO IR with input shape [1, $SEQ_LEN]"
    # Use a temp .py file so openvino's worker multiprocessing.spawn doesn't
    # try to re-import `<stdin>` (which fails noisily on every fork).
    PY_TMP="$(mktemp -t fetch-embed-XXXX.py)"
    cat > "$PY_TMP" <<'PY'
import sys
import openvino as ov
from openvino import PartialShape

src, dst, seq = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = ov.convert_model(src)
m.reshape({inp.any_name: PartialShape([1, seq]) for inp in m.inputs})
ov.save_model(m, dst)
print(f"✓ wrote {dst} + .bin (seq_len={seq})")
PY
    python3 "$PY_TMP" "$ONNX_SRC" "$IR_XML" "$SEQ_LEN"
    rm -f "$PY_TMP"
else
    echo "✓ $IR_XML / $IR_BIN already up to date"
fi

echo
echo "done. ready to run:"
ls -lh "$IR_XML" "$IR_BIN" "$TOKENIZER"
