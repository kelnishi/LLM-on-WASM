#!/usr/bin/env bash
# Example: run a GGUF LLM through WasmEdge with the wasi-nn-ggml plugin.
#
# Runtime:   WasmEdge (https://wasmedge.org) — install with:
#                curl -sSf \
#                  https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh \
#                | bash
#            then `source ~/.wasmedge/env` (already in your shell rc if
#            you ran the installer).
#
# Guest:     guest-llm-witx (wasm32-wasip1 target, wasmedge-wasi-nn
#            crate). Different binary from the wasm32-wasip2 guest-llm
#            we ship to WACS — WasmEdge expects the legacy WITX
#            (Preview 1) ABI; WACS speaks both but the wasm guest has
#            to commit at compile time.
#
# Model:     Qwen2.5 0.5B Instruct Q4_K_M GGUF (same file run-llm.sh
#            uses against the LlamaSharp backend on WACS).
#
# WasmEdge registers the GGUF under a name via --nn-preload; the
# guest's `load_by_name(MODEL_NAME)` resolves to it.

set -euo pipefail
cd "$(dirname "$0")/.."

REPO_ROOT="$(pwd)"
WASM="$REPO_ROOT/target/wasm32-wasip1/release/wasi-nn-llm-witx.wasm"
MODEL_DIR="$REPO_ROOT/models"
MODEL_NAME="qwen2.5-0.5b-instruct-q4_k_m"
GGUF="$MODEL_DIR/$MODEL_NAME.gguf"
# The host's pre-registered name the guest's load_by_name(...) looks
# for. Distinct from the on-disk filename: the --nn-preload syntax
# is `<registration-name>:GGML:AUTO:<path>`.
PRELOAD_NAME="default"

STDERR_REDIRECT="/dev/null"
case "${1:-}" in
    -v|--verbose) STDERR_REDIRECT="/dev/stderr" ;;
esac

# Build the guest if it isn't already. Different cargo target than
# the wasi-p2 harnesses.
if [ ! -f "$WASM" ]; then
    echo "→ building guest-llm-witx (wasm32-wasip1 release)…" >&2
    cargo build -p wasi-nn-llm-witx --target wasm32-wasip1 --release
fi

# WasmEdge isn't on PATH by default — install puts it at ~/.wasmedge.
if ! command -v wasmedge >/dev/null 2>&1; then
    if [ -f "$HOME/.wasmedge/env" ]; then
        # WasmEdge's env script touches unset vars (DYLD_LIBRARY_PATH,
        # etc.); briefly relax `set -u` while sourcing it.
        set +u
        # shellcheck disable=SC1091
        source "$HOME/.wasmedge/env"
        set -u
    fi
fi
command -v wasmedge >/dev/null 2>&1 || {
    echo "error: \`wasmedge\` not on PATH. install with:" >&2
    echo "        curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash" >&2
    echo "        then: source ~/.wasmedge/env" >&2
    exit 1
}

[ -f "$GGUF" ] || {
    echo "error: $GGUF not found. run scripts/fetch-gguf.sh first." >&2
    exit 1
}

# Forward MODEL_NAME so the guest knows which pre-registered name
# to ask for; the guest's default is "default" anyway, so this is
# only needed if you preload a different name.
export MODEL_NAME="$PRELOAD_NAME"

# Invocation:
#
#   --nn-preload <name>:<encoding>:<target>:<path>
#                         registers a model with WasmEdge's wasi-nn-ggml
#                         plugin under <name>; the guest's
#                         `load_by_name(<name>)` resolves to it
#
# stdin / stdout pass through as usual; stderr carries WasmEdge's
# llama.cpp chatter (model load, KV cache layout, etc.) and is
# /dev/null'd by default — pass -v to see it.
exec wasmedge \
    --nn-preload "$PRELOAD_NAME:GGML:AUTO:$GGUF" \
    "$WASM" \
    2> "$STDERR_REDIRECT"
