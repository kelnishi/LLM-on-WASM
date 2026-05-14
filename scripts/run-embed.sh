#!/usr/bin/env bash
# Example: semantic-search REPL against MiniLM-L6-v2 through WACS with
# the OpenVINO backend.
#
# Backend:   WACS.WASI.NN.OpenVino (--bind to the staged backend dll)
# Model:     all-MiniLM-L6-v2 in OpenVINO IR format. The fetch script
#            converts the HuggingFace ONNX with `[1, 64]` input shape
#            pinned (wasi-nn requires concrete dims).
#
# The guest builds 3 I64 tensors (input_ids / attention_mask /
# token_type_ids), runs ctx.compute, mean-pools the [1, 64, 384] FP32
# output over the attention mask, and ranks a built-in corpus by
# cosine similarity against each typed query.
#
# Setup once:  scripts/setup.sh             (wacs + backend NuGets)
#              scripts/fetch-embed-model.sh (downloads MiniLM,
#                                            converts ONNX → IR)
# Run:         scripts/run-embed.sh         (this script)
# Verbose:     scripts/run-embed.sh -v      (OpenVINO load chatter)

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
WASM="$REPO_ROOT/target/wasm32-wasip2/release/wasi-nn-embed.wasm"
MODEL_DIR="$REPO_ROOT/models"
BACKEND_DLL="$REPO_ROOT/tools/Backends/bin/Release/net8.0/Wacs.WASI.NN.OpenVino.dll"

STDERR_REDIRECT="/dev/null"
case "${1:-}" in
    -v|--verbose) STDERR_REDIRECT="/dev/stderr" ;;
esac

# Build the guest if it isn't already.
if [ ! -f "$WASM" ]; then
    echo "→ building guest-embed (wasm32-wasip2 release)…" >&2
    cargo build -p wasi-nn-embed --target wasm32-wasip2 --release
fi

# Sanity checks.
command -v wacs >/dev/null 2>&1 || {
    echo "error: \`wacs\` not on PATH. run scripts/setup.sh first." >&2
    exit 1
}
[ -f "$BACKEND_DLL" ] || {
    echo "error: $BACKEND_DLL not found. run scripts/setup.sh first." >&2
    exit 1
}
for f in minilm.xml minilm.bin sentence-tokenizer.json; do
    [ -f "$MODEL_DIR/$f" ] || {
        echo "error: $MODEL_DIR/$f not found. run scripts/fetch-embed-model.sh first." >&2
        exit 1
    }
done

# Invocation:
#
#   --wasip2              enable WASI Preview 2 component-model dispatch
#   --bind BACKEND_DLL    load OpenVino as an IBindable host package
#   -d models             preopen models/ as /models for the guest
exec wacs run "$WASM" \
    --wasip2 \
    --bind "$BACKEND_DLL" \
    -d models \
    2> "$STDERR_REDIRECT"
