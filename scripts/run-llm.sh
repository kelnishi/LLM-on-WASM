#!/usr/bin/env bash
# Launch the wasi-nn load-by-name harness through WACS.
#
# Defaults wire the LlamaSharp backend + GGUF model directory:
#   - WACS_WASINN_GGUF_DIR = ./models
#   - --bind Wacs.WASI.NN.LlamaSharp
#   - MODEL_NAME = qwen2.5-0.5b-instruct-q4_k_m (compiled-in default)
#
# The same wasm guest works against any load-by-name backend
# (LlamaSharp / OnnxRuntimeGenAI / future) — pick by overriding
# the --bind backend (BACKEND_DLL env var) and the in-guest model
# stem (MODEL_NAME env var, forwarded to the guest via -e).
#
#   stdout — REPL prompts + model replies (always shown)
#   stderr — backend native chatter + harness status eprintlns
#            (suppressed by default — see flags below)
#
# Stdin passes through unchanged; the harness is interactive by
# default. Pipe stdin for non-interactive runs:
#     echo -e "Hello\n/bye" | scripts/run-llm.sh
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
#   `wacs` on PATH and backend NuGets staged — one-shot setup via:
#       scripts/setup.sh
#
# Overrides (env vars):
#   WACS         wacs invocation (default: `wacs`); override to e.g.
#                `dotnet wacs` for a local-tool-manifest setup
#   BACKENDS_DIR directory of staged backend dlls
#                (default: tools/Backends/bin/Release/net8.0)
#   BACKEND_DLL  full path to the wasi-nn backend .dll
#                (default: $BACKENDS_DIR/Wacs.WASI.NN.LlamaSharp.dll)
#   MODEL_DIR    directory of model files (default: ./models)
#                forwarded to the host as both
#                WACS_WASINN_GGUF_DIR and WACS_WASINN_GENAI_DIR
#                so the backend picks whichever one it scans for
#   MODEL_NAME   model stem the guest calls load_by_name() with
#                (default: in-guest DEFAULT_MODEL_NAME constant)
#   WASM         full path to the guest wasm component
#                (default: target/wasm32-wasip2/release/wasi-nn-llm.wasm)

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
: "${BACKENDS_DIR:=$REPO_ROOT/tools/Backends/bin/Release/net8.0}"
: "${BACKEND_DLL:=$BACKENDS_DIR/Wacs.WASI.NN.LlamaSharp.dll}"
: "${MODEL_DIR:=$REPO_ROOT/models}"
: "${WASM:=$REPO_ROOT/target/wasm32-wasip2/release/wasi-nn-llm.wasm}"

# Build the guest if missing — fast (release already incremental).
if [ ! -f "$WASM" ]; then
    echo "→ building guest-llm (wasm32-wasip2 release)…" >&2
    cargo build -p wasi-nn-llm --target wasm32-wasip2 --release
fi

# Sanity checks (these errors go to the user's stderr regardless of mode).
command -v $WACS >/dev/null 2>&1 || {
    echo "error: WACS CLI ($WACS) not found on PATH" >&2
    echo "        install it with: dotnet tool install --global WACS.Cli" >&2
    echo "        or set WACS=/path/to/wacs" >&2
    exit 1
}

[ -f "$BACKEND_DLL" ] || {
    echo "error: backend DLL not found at $BACKEND_DLL" >&2
    echo "        stage the backend NuGets with: scripts/setup.sh" >&2
    echo "        or set BACKEND_DLL=/path/to/Wacs.WASI.NN.<backend>.dll" >&2
    exit 1
}

# A model dir miss isn't strictly an error here — the backend
# decides what kinds of files (and which env var) it scans. Pass
# both common env-var names so LlamaSharp + OnnxRuntimeGenAI
# both find their model directory without per-backend script
# variants.
export WACS_WASINN_GGUF_DIR="$MODEL_DIR"
export WACS_WASINN_GENAI_DIR="$MODEL_DIR"

# Forward MODEL_NAME and the env vars the backend's IBindable
# reads into the wasm guest's environment (WASI only exposes
# explicitly-listed env vars). Skip if MODEL_NAME isn't set —
# the guest falls back to its compiled-in DEFAULT_MODEL_NAME.
ENV_ARGS=()
[ -n "${MODEL_NAME:-}" ] && ENV_ARGS+=(-e "MODEL_NAME=$MODEL_NAME")

case "$STDERR_MODE" in
    quiet)
        exec $WACS run "$WASM" --wasip2 --bind "$BACKEND_DLL" \
            "${ENV_ARGS[@]}" 2>/dev/null ;;
    passthrough)
        exec $WACS run "$WASM" --wasip2 --bind "$BACKEND_DLL" \
            "${ENV_ARGS[@]}" ;;
    log)
        : > "$LOG_FILE"  # truncate
        echo "logging stderr to $LOG_FILE" >&2
        exec $WACS run "$WASM" --wasip2 --bind "$BACKEND_DLL" \
            "${ENV_ARGS[@]}" 2>>"$LOG_FILE" ;;
esac
