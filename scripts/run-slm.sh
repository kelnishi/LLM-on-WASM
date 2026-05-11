#!/usr/bin/env bash
# Launch the wasi-nn ONNX SLM harness (Gemma 3 270M) through WACS.
#
# Uses --wasi-nn (bundled OnnxRuntime backend) + --native-memory
# (the 1.14 GB ONNX byte-load crosses the 2 GiB ManagedArray cap)
# + -d models::/models (the guest does std::fs::read("/models/...")).
#
#   stdout — REPL prompts + model replies (always shown)
#   stderr — guest status eprintlns + ORT diagnostics
#            (suppressed by default — see flags below)
#
# Stdin passes through unchanged; the harness is interactive by
# default. Pipe stdin for non-interactive runs:
#     echo -e "What is 2+2?\n/bye" | scripts/run-slm.sh
#
# Flags:
#   -v, --verbose         pass stderr through to the terminal
#       --log <file>      capture stderr to <file>
#   -h, --help            this message
#
# Without -v or --log, stderr is redirected to /dev/null so the
# terminal carries only the REPL.
#
# Requires:
#   `wacs` on PATH — install with:
#       dotnet tool install --global WACS.Cli
#
# Overrides (env vars):
#   WACS        wacs invocation (default: `wacs`); override to e.g.
#               `dotnet wacs` for a local-tool-manifest setup
#   MODEL_DIR   directory containing gemma3_270m.onnx + tokenizer.json
#               (default: ./models)
#   WASM        full path to the guest wasm component
#               (default: target/wasm32-wasip2/release/wasi-nn-slm.wasm)

set -euo pipefail

STDERR_MODE="quiet"
LOG_FILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        -v|--verbose)
            STDERR_MODE="passthrough"; shift ;;
        --log)
            [ $# -ge 2 ] || { echo "error: --log requires a path" >&2; exit 2; }
            STDERR_MODE="log"; LOG_FILE="$2"; shift 2 ;;
        --log=*)
            STDERR_MODE="log"; LOG_FILE="${1#--log=}"; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)
            echo "error: unknown argument: $1" >&2
            echo "       run \`$0 --help\` for usage" >&2
            exit 2 ;;
    esac
done

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

: "${WACS:=wacs}"
: "${MODEL_DIR:=$REPO_ROOT/models}"
: "${WASM:=$REPO_ROOT/target/wasm32-wasip2/release/wasi-nn-slm.wasm}"

# Build the guest if missing — fast (release already incremental).
if [ ! -f "$WASM" ]; then
    echo "→ building guest (wasm32-wasip2 release)…" >&2
    cargo build -p wasi-nn-slm --target wasm32-wasip2 --release
fi

# Sanity checks (these errors go to the user's stderr regardless of mode).
command -v $WACS >/dev/null 2>&1 || {
    echo "error: WACS CLI ($WACS) not found on PATH" >&2
    echo "        install it with: dotnet tool install --global WACS.Cli" >&2
    echo "        or set WACS=/path/to/wacs" >&2
    exit 1
}

[ -f "$MODEL_DIR/gemma3_270m.onnx" ] || {
    echo "error: $MODEL_DIR/gemma3_270m.onnx not found" >&2
    echo "        fetch it with: scripts/fetch-model.sh" >&2
    echo "        or set MODEL_DIR=/path/to/onnx-dir" >&2
    exit 1
}

[ -f "$MODEL_DIR/tokenizer.json" ] || {
    echo "error: $MODEL_DIR/tokenizer.json not found" >&2
    echo "        fetch it with: scripts/fetch-model.sh" >&2
    exit 1
}

case "$STDERR_MODE" in
    quiet)
        exec $WACS run "$WASM" \
            --wasip2 --wasi-nn --native-memory -d "$MODEL_DIR::/models" \
            2>/dev/null ;;
    passthrough)
        exec $WACS run "$WASM" \
            --wasip2 --wasi-nn --native-memory -d "$MODEL_DIR::/models" ;;
    log)
        : > "$LOG_FILE"  # truncate
        echo "logging stderr to $LOG_FILE" >&2
        exec $WACS run "$WASM" \
            --wasip2 --wasi-nn --native-memory -d "$MODEL_DIR::/models" \
            2>>"$LOG_FILE" ;;
esac
