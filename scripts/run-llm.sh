#!/usr/bin/env bash
# Example: run a GGUF LLM (Qwen2.5 0.5B Instruct Q4_K_M) through WACS
# with the LlamaSharp backend.
#
# Backend:   WACS.WASI.NN.LlamaSharp (--bind to the staged backend dll)
# Model:     single .gguf file resolved by name. LlamaSharp does the
#            whole pipeline host-side — tokenization, chat templating,
#            KV-cached decode, sampling. The wasm guest just shuttles
#            UTF-8 prompt bytes in and UTF-8 reply bytes out.
#
# Setup once:    scripts/setup.sh           (wacs + backend NuGets)
#                scripts/fetch-gguf.sh      (downloads the GGUF)
# Run:           scripts/run-llm.sh         (this script)
# Verbose mode:  scripts/run-llm.sh -v      (show backend chatter on stderr)
#
# Type a prompt and press enter; `/bye` to exit.

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
WASM="$REPO_ROOT/target/wasm32-wasip2/release/wasi-nn-llm.wasm"
MODEL_DIR="$REPO_ROOT/models"
BACKEND_DLL="$REPO_ROOT/tools/Backends/bin/Release/net8.0/Wacs.WASI.NN.LlamaSharp.dll"
MODEL_NAME="qwen2.5-0.5b-instruct-q4_k_m"

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
[ -f "$MODEL_DIR/$MODEL_NAME.gguf" ] || {
    echo "error: $MODEL_DIR/$MODEL_NAME.gguf not found. run scripts/fetch-gguf.sh first." >&2
    exit 1
}

# Env vars exported here flow through to the wasm guest's WASI
# Preview 2 environment automatically (wacs forwards the host
# process env on the --wasip2 dispatch path).
#
#   WACS_WASINN_GGUF_DIR  consumed by LlamaSharp's IBindable; scans
#                         for *.gguf and registers each under its
#                         filename stem
#   MODEL_NAME            read by the guest's main() — picks which
#                         stem load_by_name() asks for
export WACS_WASINN_GGUF_DIR="$MODEL_DIR"
export MODEL_NAME

# Invocation:
#
#   --wasip2              enable WASI Preview 2 component-model dispatch
#   --bind BACKEND_DLL    load LlamaSharp as an IBindable host package
exec wacs run "$WASM" \
    --wasip2 \
    --bind "$BACKEND_DLL" \
    2> "$STDERR_REDIRECT"
