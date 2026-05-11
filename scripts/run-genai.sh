#!/usr/bin/env bash
# Launch the wasi-nn load-by-name harness against the
# Wacs.WASI.NN.OnnxRuntimeGenAI backend.
#
# Targets generative LLMs in onnxruntime-genai format — i.e.,
# model directories with the layout:
#
#   <MODEL_NAME>/
#       genai_config.json
#       tokenizer.json
#       tokenizer_config.json
#       special_tokens_map.json
#       model.onnx
#       model.onnx.data     (external weights, if used)
#
# Drop one or more of these under $MODEL_DIR (default ./models)
# and set MODEL_NAME to the directory's basename — the host
# scans $WACS_WASINN_GENAI_DIR for any subdir containing
# `genai_config.json` and registers each under its directory
# name. The guest's `graph.load-by-name(...)` then resolves.
#
# Same compiled wasm as run-llm.sh (LlamaSharp); just a
# different backend DLL + model directory layout.
#
# Defaults:
#   BACKEND_DLL → Wacs.WASI.NN.OnnxRuntimeGenAI.dll staged by setup.sh
#   MODEL_NAME  → gemma-3-270m-it-genai
#                 (e.g., smartvest-llc/gemma-3-270m-it-genai from HF)
#
# All other env vars, flags (-v / --log / -h), and behavior are
# delegated to run-llm.sh — see `scripts/run-llm.sh --help`.

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"

: "${BACKENDS_DIR:=$REPO_ROOT/tools/Backends/bin/Release/net8.0}"
: "${BACKEND_DLL:=$BACKENDS_DIR/Wacs.WASI.NN.OnnxRuntimeGenAI.dll}"
: "${MODEL_NAME:=gemma-3-270m-it-genai}"

export BACKENDS_DIR BACKEND_DLL MODEL_NAME

exec "$REPO_ROOT/scripts/run-llm.sh" "$@"
