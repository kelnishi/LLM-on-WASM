# Run an LLM in WebAssembly on WACS

A showcase of running real LLMs and small ML models inside a
WebAssembly sandbox using [**WACS**](https://github.com/kelnishi/WACS) —
a pure-.NET wasm runtime with first-class
[`wasi-nn`](https://github.com/WebAssembly/wasi-nn) support. The wasm
guests are stock `cargo build --target wasm32-wasip2` components, well
under 100 lines of Rust apiece; WACS does the rest.

> [WACS on GitHub](https://github.com/kelnishi/WACS) ·
> [Wacs.WASI.NN backend packages](https://www.nuget.org/packages?q=Wacs.WASI.NN) ·
> [Discussions](https://github.com/kelnishi/WACS/discussions)

## Why this matters

A portable wasm component holds the orchestration — prompt I/O, REPL
loop, error handling — and WACS's [wasi-nn](https://github.com/WebAssembly/wasi-nn)
host binds it to a real inference backend at run time. Swap the
backend, swap the model, change runtime entirely — the guest doesn't
recompile. That's the wasm sandbox doing what wasm sandboxes are
supposed to do, and WACS makes it a one-liner.

```
   ┌────────────────────────────────────────┐
   │  wasm guest (wasm32-wasip2 component)  │   ~60 lines Rust each
   │     stdin / stdout REPL, wasi-nn       │   compiled once,
   │     load_by_name → set_input →         │   no native deps
   │     compute → get_output               │
   └──────────────────┬─────────────────────┘
                      │  wasi:nn/... @ 0.2.0-rc-2024-10-28
   ┌──────────────────┴─────────────────────┐
   │  WACS — pure-.NET wasm runtime         │
   │     wasi-p2 (WIT) + wasi-p1 (WITX)     │
   │     pluggable IBindable backends       │
   └──────────────────┬─────────────────────┘
                      │  --bind WACS.WASI.NN.<Backend>.dll
   ┌──────────────────┴─────────────────────┐
   │  Backend NuGets (host-side acceleration)│
   │    LlamaSharp  → llama.cpp + Metal/CUDA│
   │    OnnxRuntime → ONNX Runtime          │
   │    OnnxRuntimeGenAI → KV-cached SLMs   │
   │    TorchSharp  → libtorch              │
   │    ML.NET      → classical ML pipelines│
   └────────────────────────────────────────┘
```

## What WACS brings

- **Pure-.NET wasm runtime** — no native runtime dependency, ships as
  a `dotnet tool` (`WACS.Cli`) on any platform .NET runs on. Embed it
  in any .NET host with a `WasmRuntime`; no FFI glue.
- **First-class wasi-nn** across **five** backends today
  (LlamaSharp / llama.cpp, ONNX Runtime, OnnxRuntime GenAI, TorchSharp /
  libtorch, ML.NET) — each shipped as a separate NuGet so an embedder
  only pulls in what they need.
- **Both wasi-nn ABI flavors**: the modern WIT/component-model
  interface (`wasi:nn/...@0.2.0-rc-2024-10-28`) for `wasm32-wasip2`
  guests, plus the legacy WITX (`wasi_ephemeral_nn`) ABI for Preview 1
  guests — making the same .gguf / .onnx asset accessible to both
  WACS and WasmEdge with no guest-side changes.
- **Hardware-accelerated host-side** — Metal on Apple Silicon, CUDA on
  Linux/Windows GPUs, AVX/NEON CPU paths — picked up automatically by
  the backend NuGets. Guests stay portable wasm bytecode.
- **`--bind <Backend.dll>` UX** — drop a backend dll on the command
  line and WACS isolates its native deps in a per-backend
  `AssemblyLoadContext`. Multiple backends co-exist without `LD_*`
  juggling.

## How the examples use it

The guests target `wasm32-wasip2` (the WASI Preview 2 component model)
and use the modern wit-level wasi-nn interface
(`wasi:nn/...@0.2.0-rc-2024-10-28`). All the heavy lifting
(tokenization, KV cache, sampling, chat templating) lives host-side in
WACS's backend NuGet packages. The guests just shuttle prompt bytes
in and reply bytes (or tensors) out through the wasi-nn ABI — and
they're identical across backends. One `guest-llm.wasm` runs against
either LlamaSharp (GGUF) or OnnxRuntime GenAI (ONNX SLM) just by
switching which backend dll `--bind` points at.

## What's included

| Example | Guest | Model | wasi-nn backend |
|---|---|---|---|
| ONNX SLM | `guest/` | Gemma 3 270M (`.onnx`, FP32 ~1.14 GB) | `WACS.WASI.NN.OnnxRuntime` (byte-loaded) |
| GGUF LLM | `guest-llm/` | Qwen2.5 0.5B Instruct (`.gguf`, Q4_K_M ~352 MB) | `WACS.WASI.NN.LlamaSharp` (llama.cpp) |
| ONNX LLM (GenAI) | `guest-llm/` (same wasm) | Gemma 3 270M Instruct (GenAI format dir, ~864 MB) | `WACS.WASI.NN.OnnxRuntimeGenAI` |
| TorchScript | `guest-torch/` | XOR MLP (`.pt`, ~6 KB) | `WACS.WASI.NN.TorchSharp` (libtorch) |

A legacy example (`guest-llm-witx/`) targets WASI Preview 1's older
`wasi_ephemeral_nn` ABI — kept for interoperability with WasmEdge
and other Preview 1 hosts. See [Legacy: WASI Preview 1](#legacy-wasi-preview-1)
at the end.

## Setup

Prerequisites:

- [.NET SDK 8 or 9](https://dotnet.microsoft.com/download) — `dotnet` on PATH
- [Rust + Cargo](https://rustup.rs) — `cargo` on PATH. The
  `wasm32-wasip2` and `wasm32-wasip1` targets are pinned in
  `rust-toolchain.toml` and installed automatically on first build.
- `python3` with `torch` — only needed for the XOR MLP example
  (used to train and trace a TorchScript module).
- ~3 GB free disk for the model downloads.

One-shot setup:

```sh
./scripts/setup.sh
```

This:

1. Installs `WACS.Cli` as a .NET global tool — `wacs` lands on your
   PATH.
2. Stages every `WACS.WASI.NN.*` backend NuGet (managed dlls plus
   RID-specific native libs like `libtorch`, `libllama`,
   `libonnxruntime`) into `tools/Backends/bin/Release/net8.0/`. No
   source code is compiled — MSBuild runs the NuGet restore +
   native-dep staging pipeline against an empty csproj.

Re-run `setup.sh` if you change the pinned backend versions in
`tools/Backends/Backends.csproj`.

## Running the examples

Each `scripts/run-*.sh` is a self-contained launcher. They build the
guest wasm on first run if needed, sanity-check that the model file
is present, and `exec wacs run` with the right flags for the backend.

Stderr is silenced to `/dev/null` by default so the REPL stays clean.
Pass `-v` to see backend chatter (model load progress, KV-cache
layout, native-library version banners, etc.):

```sh
./scripts/run-llm.sh -v
```

### ONNX SLM — Gemma 3 270M

```sh
./scripts/fetch-model.sh            # ~1.14 GB ONNX + tokenizer
./scripts/run-slm.sh
```

```
>>> What is 2+2?
2 + 2 = 4
>>> /bye
```

The guest reads `gemma3_270m.onnx` from a preopened directory,
tokenizes the prompt with the in-guest HuggingFace `tokenizers` crate,
runs greedy generation through wasi-nn, and decodes the reply.

`run-slm.sh` passes `--native-memory` to lift WACS's default 2 GiB
linear-memory cap — the 1.14 GB model byte-buffer transit needs the
full 4 GiB wasm32 ceiling.

### GGUF LLM — Qwen2.5 0.5B via llama.cpp

```sh
./scripts/fetch-gguf.sh             # ~352 MB GGUF
./scripts/run-llm.sh
```

```
>>> What is 2+2?
 2 + 2 equals 4. …
>>> /bye
```

`LlamaSharp` does the whole pipeline host-side — tokenization, chat
templating, KV-cached decode, sampling, EOS detection. The guest
sends a UTF-8 prompt as a single U8 tensor named `"prompt"` and
reads the UTF-8 reply back from a single U8 output tensor. ~70 lines
of guest code, no LLM-specific machinery anywhere in the wasm.

### ONNX LLM (GenAI format) — Gemma 3 270M

```sh
# Download a GenAI-format model directory under ./models/:
huggingface-cli download smartvest-llc/gemma-3-270m-it-genai \
    --local-dir models/gemma-3-270m-it-genai
./scripts/run-genai.sh
```

```
>>> What is 2+2?
2 + 2 = 4
>>> /bye
```

Uses Microsoft's `Microsoft.ML.OnnxRuntimeGenAI` library under the
hood — first-class tokenizer + KV cache + sampling for ONNX-format
LLMs (Gemma 3 / Llama 3 / Qwen 2.5 / Phi 4). Same ergonomics as the
LlamaSharp track.

A "GenAI-format" model is a directory containing `genai_config.json`,
`tokenizer.json` + sibling config files, `model.onnx`, and
`model.onnx.data` for external weights. Either download a pre-built
one from HuggingFace (as above) or build your own with
[`onnxruntime-genai`'s model_builder](https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models).

**The same compiled wasm runs against both LlamaSharp and
OnnxRuntimeGenAI.** Only the `BACKEND_DLL` and `MODEL_NAME`
environment variables differ — the guest's wire shape is
backend-agnostic.

### TorchScript — XOR MLP

```sh
./scripts/build-xor-mlp.sh          # trains + traces, ~6 KB output
WACS_WASINN_TORCH_DIR=$(pwd)/models \
  wacs run target/wasm32-wasip2/release/wasi-nn-torch.wasm \
      --wasip2 \
      --bind tools/Backends/bin/Release/net8.0/Wacs.WASI.NN.TorchSharp.dll
```

```
  XOR(0, 0) -> sigmoid=0.0000  pred=0  expected=0  OK
  XOR(0, 1) -> sigmoid=1.0000  pred=1  expected=1  OK
  XOR(1, 0) -> sigmoid=0.9994  pred=1  expected=1  OK
  XOR(1, 1) -> sigmoid=0.0000  pred=0  expected=0  OK

all cases pass
```

A 2-layer MLP trained for XOR, scripted to TorchScript, then
exercised through wasi-nn against libtorch. Tiny end-to-end demo of
the FP32 tensor I/O path without any LLM-class machinery — verifies
that the wasi-nn → TorchSharp wiring works for a model that just
takes two floats and returns one float.

(`scripts/build-xor-mlp.sh` requires `python3` and `pip install torch`.)

## Legacy: WASI Preview 1

Before the WASI Preview 2 component model existed, `wasi-nn` shipped
as a Preview 1 core-module interface called `wasi_ephemeral_nn` — the
**witx** flavor. It's still the only flavor supported by some
runtimes ([WasmEdge](https://wasmedge.org) being the most prominent)
and by older Rust toolchains.

The witx ABI is functionally the same surface — `load_by_name`,
`init_execution_context`, `set_input`, `compute`, `get_output` — but
exposed as flat core-module imports instead of component-level
resources, with manual handle management and stateful per-context
input/output slots.

WACS supports both ABIs simultaneously, so picking witx is purely a
toolchain decision (older `wasi-nn` crates, older Rust targets). The
prototypical examples in this repo use the modern wit/component path.

The repo ships one witx-targeted guest (`guest-llm-witx/`, a
`wasm32-wasip1` build of the load-by-name REPL) primarily for
running against WasmEdge as a cross-runtime sanity check:

```sh
# Install WasmEdge with the wasi-nn-ggml plugin:
curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install_v2.sh | bash
source ~/.wasmedge/env

./scripts/fetch-gguf.sh             # if you haven't already
./scripts/run-llm-wasmedge.sh
```

```
>>> What is 2+2?
 The sum of 2 and 2 is 4. …
>>> /bye
```

For new projects, prefer the wit/component model path — that's where
the broader wasi-nn ecosystem is heading.

## How it works

Each example follows the same three-piece shape:

```
                  wasm guest (this repo)
                    │
                    │  wasi-nn ABI
                    ▼
                  wasm runtime (WACS / WasmEdge)
                    │
                    │  IBackend SPI (WACS) / plugin C ABI (WasmEdge)
                    ▼
                  backend (WACS.WASI.NN.* NuGet,
                  WasmEdge wasi-nn-ggml plugin, …)
                    │
                    │  P/Invoke / FFI
                    ▼
                  native library (libtorch, libllama,
                  libonnxruntime, onnxruntime-genai)
```

The wasm guest only knows about `wasi-nn`. It calls `load_by_name`,
gets a graph handle, feeds tensors via `compute(…)`, reads outputs
back. Backend selection happens entirely outside the wasm sandbox
— WACS dispatches by the encoding the guest requests plus whichever
IBindable was wired via `--bind`.

The same compiled wasm is portable across:

- **Backends** — swap `--bind <X.dll>` for a different
  `WACS.WASI.NN.*` package. The guest is encoding-agnostic at the
  wire level for the load-by-name examples; `guest-llm/` works
  against LlamaSharp, OnnxRuntimeGenAI, or any future load-by-name
  backend that follows the WasmEdge GGUF convention (U8 prompt in,
  U8 reply out).
- **Runtimes** — wasi-nn is a public spec, and any runtime that
  speaks the same ABI flavor can host the same wasm. For modern
  wit-level guests, that's WACS today; other runtimes are expected
  to grow component-model wasi-nn support. The legacy witx flavor
  also runs on WasmEdge (see [Legacy](#legacy-wasi-preview-1)).

## Repository layout

```
guest/                  Gemma 3 ONNX SLM (wasi-p2, byte-load, in-guest tokenizer)
guest-llm/              backend-agnostic load-by-name LLM REPL (wasi-p2)
guest-torch/            XOR MLP TorchScript exerciser (wasi-p2)
guest-llm-witx/         legacy WASI Preview 1 load-by-name REPL (for WasmEdge)
host/                   wasmtime + custom Rust wasi-nn backend (parity reference)
scripts/
    setup.sh            install wacs + stage backend NuGets
    fetch-model.sh      download Gemma 3 270M ONNX
    fetch-gguf.sh       download Qwen2.5 0.5B GGUF
    build-xor-mlp.sh    train + trace XOR MLP via PyTorch
    run-slm.sh          run guest/ via WACS + OnnxRuntime
    run-llm.sh          run guest-llm/ via WACS + LlamaSharp
    run-genai.sh        run guest-llm/ via WACS + OnnxRuntimeGenAI
    run-llm-wasmedge.sh run guest-llm-witx/ via WasmEdge + wasi-nn-ggml
tools/Backends/         no-source csproj that stages backend NuGets
models/                 (gitignored) downloaded models land here
```

`host/` is a small wasmtime-embedding Rust binary with a custom
wasi-nn backend that supports I64 / Fp16 / I32 / U8 tensors (the
backend that ships in upstream `wasmtime-wasi-nn` is FP32-only). It
runs `guest/` end-to-end through wasmtime instead of WACS as a parity
reference. Not needed for the main demo path but useful as a
"~300 lines of Rust showing what a minimal wasi-nn host looks like."

## References

- [WACS](https://github.com/kelnishi/WACS) — the wasm runtime under test
- [wasi-nn](https://github.com/WebAssembly/wasi-nn) — the wit / witx specs
- [wit-bindgen](https://github.com/bytecodealliance/wit-bindgen) — the Rust → wasm-component toolchain
- [onnx-community/gemma-3-270m-it-ONNX](https://huggingface.co/onnx-community/gemma-3-270m-it-ONNX) — Gemma 3 ONNX source
- [Qwen/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF) — Qwen2.5 GGUF source
- [smartvest-llc/gemma-3-270m-it-genai](https://huggingface.co/smartvest-llc/gemma-3-270m-it-genai) — Gemma 3 GenAI-format source
