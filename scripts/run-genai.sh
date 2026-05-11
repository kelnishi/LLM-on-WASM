#!/usr/bin/env bash
# Example: run a GenAI-format ONNX LLM (Gemma 3 270M Instruct) through
# WACS with the OnnxRuntimeGenAI backend.
#
# Backend:   WACS.WASI.NN.OnnxRuntimeGenAI (--bind to the staged dll)
# Model:     directory containing genai_config.json + tokenizer.json +
#            model.onnx (+ model.onnx.data for external weights),
#            resolved by directory name. Microsoft.ML.OnnxRuntimeGenAI
#            handles tokenization, chat templating, KV-cached decode,
#            sampling — the wasm guest just shuttles UTF-8 prompt bytes
#            in and UTF-8 reply bytes out (same wire shape as the
#            LlamaSharp track; same compiled wasm).
#
# Setup once:    scripts/setup.sh           (wacs + backend NuGets)
#                # Drop a GenAI-format model directory under ./models/
#                # — e.g., download `smartvest-llc/gemma-3-270m-it-genai`
#                # from HF into models/gemma-3-270m-it-genai/
# Run:           scripts/run-genai.sh       (this script)
# Verbose mode:  scripts/run-genai.sh -v
#
# Type a prompt and press enter; `/bye` to exit.

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
WASM="$REPO_ROOT/target/wasm32-wasip2/release/wasi-nn-llm.wasm"
MODEL_DIR="$REPO_ROOT/models"
BACKEND_DLL="$REPO_ROOT/tools/Backends/bin/Release/net8.0/Wacs.WASI.NN.OnnxRuntimeGenAI.dll"
MODEL_NAME="gemma-3-270m-it-genai"

STDERR_REDIRECT="/dev/null"
case "${1:-}" in
    -v|--verbose) STDERR_REDIRECT="/dev/stderr" ;;
esac

# Build the guest if it isn't already.
if [ ! -f "$WASM" ]; then
    echo "→ building guest-llm (wasm32-wasip2 release)…" >&2
    cargo build -p wasi-nn-llm --target wasm32-wasip2 --release
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
[ -f "$MODEL_DIR/$MODEL_NAME/genai_config.json" ] || {
    echo "error: $MODEL_DIR/$MODEL_NAME/genai_config.json not found." >&2
    echo "        download a GenAI-format model into $MODEL_DIR/$MODEL_NAME/." >&2
    echo "        e.g. huggingface-cli download smartvest-llc/gemma-3-270m-it-genai \\" >&2
    echo "                 --local-dir $MODEL_DIR/$MODEL_NAME" >&2
    exit 1
}

# Env vars exported here flow through to the wasm guest's WASI
# Preview 2 environment automatically (wacs forwards the host
# process env on the --wasip2 dispatch path).
#
#   WACS_WASINN_GENAI_DIR consumed by OnnxRuntimeGenAI's IBindable;
#                         scans for subdirs with a genai_config.json
#                         and registers each under its directory name
#   MODEL_NAME            read by the guest's main() — picks which
#                         model directory load_by_name() asks for
export WACS_WASINN_GENAI_DIR="$MODEL_DIR"
export MODEL_NAME

# Invocation: same shape as run-llm.sh modulo the backend dll —
# the guest is backend-agnostic at the wire level (both backends
# speak the WasmEdge U8-in / U8-out convention).
exec wacs run "$WASM" \
    --wasip2 \
    --bind "$BACKEND_DLL" \
    2> "$STDERR_REDIRECT"
