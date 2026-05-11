#!/usr/bin/env bash
# One-shot setup for running the wasi-nn harnesses through WACS.
#
# 1. Installs WACS.Cli as a .NET global tool (idempotent — skipped if
#    already present at the pinned version).
# 2. Stages the WACS.WASI.NN.* backend NuGets into
#    tools/Backends/bin/Release/net8.0/ so `--bind <path>` invocations
#    can point at a single directory with the backend dll, all
#    managed transitive deps, and the runtimes/<rid>/native/ natives.
#
# Step 2 runs `dotnet build` on tools/Backends.csproj, which has no
# source files — MSBuild produces no compiled assembly of its own,
# just runs the NuGet restore + package-dep-staging pipeline. Look
# for the `CSC : warning CS2008: No source files specified` line in
# the build output: that confirms no actual compilation happened.
# Conceptually this is `nuget restore + nuget extract` — there's no
# bash-native equivalent of dep-staging that handles transitive
# runtime packages, so the empty-csproj trick is the path of least
# resistance.

set -euo pipefail

WACS_VERSION="${WACS_VERSION:-1.5.26}"

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

# 1. Global tool install for WACS.Cli.
if ! command -v wacs >/dev/null 2>&1; then
    echo "→ installing WACS.Cli $WACS_VERSION as a .NET global tool…"
    dotnet tool install --global WACS.Cli --version "$WACS_VERSION"
else
    # `wacs --version` exits with a non-zero code by design — the
    # output we want is on stdout regardless, so swallow status.
    installed=$( { wacs --version 2>&1 || true; } | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    if [ "$installed" = "$WACS_VERSION" ]; then
        echo "✓ WACS.Cli $installed already on PATH"
    else
        echo "→ updating WACS.Cli ($installed → $WACS_VERSION)…"
        dotnet tool update --global WACS.Cli --version "$WACS_VERSION"
    fi
fi

# 2. Stage the WASI.NN backend NuGets into tools/Backends/bin/.
echo "→ staging backend NuGets via tools/Backends/Backends.csproj…"
dotnet build "$REPO_ROOT/tools/Backends" -c Release 2>&1 | tail -3

STAGED_DIR="$REPO_ROOT/tools/Backends/bin/Release/net8.0"
echo "✓ backend dlls + runtimes staged at:"
echo "  $STAGED_DIR"
ls "$STAGED_DIR" | grep -E "Wacs\.WASI\.NN\." | head

echo
echo "ready. invocations:"
echo "  scripts/run-slm.sh     # ONNX SLM (Gemma 3 270M, byte-loaded)"
echo "  scripts/run-llm.sh     # LlamaSharp (Qwen2.5 0.5B GGUF)"
echo "  scripts/run-genai.sh   # OnnxRuntimeGenAI (Gemma 3 270M, GenAI format)"
