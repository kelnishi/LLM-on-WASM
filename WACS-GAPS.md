# WACS gap report — wasi-nn backend coverage

No open gaps. Both ABIs are end-to-end through stock guests:

- **wasi-p2 (WIT, `wasi:nn/...@0.2.0-rc-2024-10-28`)** — `guest-llm`
  (LlamaSharp / OnnxRuntimeGenAI), `guest` (OnnxRuntime),
  `guest-torch` (TorchSharp).
- **wasi-p1 (WITX, `wasi_ephemeral_nn`)** — `guest-llm-witx`
  (LlamaSharp). Same WasmEdge GGUF convention as the wasi-p2 guest:
  U8 prompt in / U8 reply out.

## Local-binary invocation (until 0.3.4 backends ship)

The WITX fix and follow-up wiring land in `WACS` commits ahead of
the last published `WACS.WASI.NN.* 0.2.x/0.3.1` NuGets. Until those
backends republish (so the transitive `WACS.WASI.NN >= 0.3.4` pulls
the fixed bindings), run against the local source build instead of
the global `WACS.Cli` tool:

```sh
WACS_LOCAL=/Users/kelvinnishikawa/wasm/WACS/Wacs.Console/Wacs.Console/bin/Release/net9.0/Wacs.Console
WACS_NN_SRC=/Users/kelvinnishikawa/wasm/WACS/Wacs.WASI/Wacs.WASI.NN

# wasi-p1 WITX path (Qwen2.5 0.5B GGUF via LlamaSharp)
WACS_WASINN_GGUF_DIR=$(pwd)/models \
  "$WACS_LOCAL" run \
    target/wasm32-wasip1/release/wasi-nn-llm-witx.wasm \
    --bind "$WACS_NN_SRC/Wacs.WASI.NN.LlamaSharp/bin/Release/net8.0/Wacs.WASI.NN.LlamaSharp.dll" \
    -e MODEL_NAME=qwen2.5-0.5b-instruct-q4_k_m
```

Two pitfalls to avoid:

- **`--bind` must point at the local-source backend dll**, not the
  one staged under `tools/Backends/bin/...`. `BindBackendLoadContext`
  isolates each backend's deps to its sibling dir, so a NuGet-staged
  `Wacs.WASI.NN.LlamaSharp.dll` brings the *old* `Wacs.WASI.NN.dll`
  (pre-fix witx bindings) regardless of which `wacs` binary
  launches. Confirm by MD5'ing `Wacs.WASI.NN.dll` next to whichever
  backend dll `--bind` resolves to.
- **`-e MODEL_NAME=...` is required for the wasi-p1 path.** Preview 1
  does not auto-forward host env vars; `-e` is the explicit channel.
  The wasi-p2 path uses host env directly, so `scripts/run-llm.sh`
  works with a plain `export`.

Once `WACS.WASI.NN.LlamaSharp >= 0.3.4` ships on NuGet, `tools/Backends`
re-stages with the fixed transitive `Wacs.WASI.NN.dll` and
`scripts/run-llm-witx.sh` can drop back to the global `wacs` tool.
