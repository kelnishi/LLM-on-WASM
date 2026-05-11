#!/usr/bin/env bash
# Example: run the ONNX SLM (Gemma 3 270M) through WACS.
#
# Backend: WACS.WASI.NN.OnnxRuntime (bundled with the CLI via --wasi-nn).
# Model:   single gemma3_270m.onnx file (~1.14 GB FP32) loaded via the
#          byte-loaded `graph.load(bytes, ONNX)` path. The guest reads
#          the file from a preopened /models directory and feeds it to
#          wasi-nn; HF tokenization, chat templating, and greedy
#          generation all run in the wasm guest.
#
# Setup once:    scripts/setup.sh           (wacs + backend NuGets)
#                scripts/fetch-model.sh     (downloads ONNX + tokenizer)
# Run:           scripts/run-slm.sh         (this script)
# Verbose mode:  scripts/run-slm.sh -v      (show backend chatter on stderr)
#
# Type a prompt and press enter; `/bye` to exit. Pipe stdin for
# non-interactive runs, e.g.:
#     echo -e "What is 2+2?\n/bye" | scripts/run-slm.sh

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
WASM="$REPO_ROOT/target/wasm32-wasip2/release/wasi-nn-slm.wasm"
MODEL_DIR="$REPO_ROOT/models"

# Default: hide backend chatter so the terminal carries only the REPL.
# Pass `-v` / `--verbose` to keep stderr visible.
STDERR_REDIRECT="/dev/null"
case "${1:-}" in
    -v|--verbose) STDERR_REDIRECT="/dev/stderr" ;;
esac

# Build the guest if it isn't already.
if [ ! -f "$WASM" ]; then
    echo "→ building guest (wasm32-wasip2 release)…" >&2
    cargo build -p wasi-nn-slm --target wasm32-wasip2 --release
fi

# Sanity checks.
command -v wacs >/dev/null 2>&1 || {
    echo "error: \`wacs\` not on PATH. run scripts/setup.sh first." >&2
    exit 1
}
[ -f "$MODEL_DIR/gemma3_270m.onnx" ] || {
    echo "error: $MODEL_DIR/gemma3_270m.onnx not found. run scripts/fetch-model.sh first." >&2
    exit 1
}

# Invocation:
#
#   --wasip2          enable WASI Preview 2 component-model dispatch
#   --wasi-nn         load the bundled OnnxRuntime backend
#   --native-memory   give the guest >2 GiB of linear memory (the 1.14 GB
#                     model file crosses the default ManagedArray cap
#                     when the ONNX byte-buffer transits the canonical
#                     ABI)
#   -d MODEL_DIR::/models
#                     preopen the on-disk MODEL_DIR as /models inside the
#                     guest's WASI sandbox; the guest does
#                     std::fs::read("/models/gemma3_270m.onnx")
exec wacs run "$WASM" \
    --wasip2 \
    --wasi-nn \
    --native-memory \
    -d "$MODEL_DIR::/models" \
    2> "$STDERR_REDIRECT"
