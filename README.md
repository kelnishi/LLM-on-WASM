# wasi-nn-slm

A minimal stub that runs **Gemma 3 270M (ONNX)** inside a `wasm32-wasip2`
component, using **wasi-nn** for inference and the HuggingFace `tokenizers`
crate (with the `unstable_wasm` feature) for tokenization. Exposes a small
Ollama-style stdin REPL.

The point is portability: the same `.wasm` should run under any wasi-p2
runtime that implements wasi-nn's WIT interface with an ONNX backend.

## Layout

    Cargo.toml          — wit-bindgen + tokenizers (unstable_wasm)
    rust-toolchain.toml — pinned to wasm32-wasip2
    wit/wasi-nn.wit     — vendored from WebAssembly/wasi-nn (0.2.0-rc-2024-10-28)
    src/main.rs         — REPL + chat-template render + greedy generation loop
    scripts/fetch-model.sh — pulls model + tokenizer, inlines external weights

## Architecture notes

- **No KV cache.** Every step retokenises and re-feeds the entire prompt with
  empty `past_key_values.{i}.key/value` tensors (shape `[1, 1, 0, 256]`,
  zero bytes). Quadratic and slow but keeps the stub small. Outputs `present.*`
  are ignored.
- **External data is inlined at fetch time.** The wasi-nn `graph::load` ABI
  takes raw bytes; ONNX `external_data` references would never resolve from
  the sandbox, so `fetch-model.sh` calls `onnx.save(..., save_as_external_data=False)`
  to merge the weights into a single self-contained `gemma3_270m.onnx`.
- **Chat template** is hand-rolled rather than running Jinja in wasm — the
  Gemma format is small enough that it fits in `render_chat()`.
- **Streaming** is done by re-decoding all generated tokens after each step
  and printing the byte-suffix at safe UTF-8 boundaries.

## Build

```sh
cargo build --target wasm32-wasip2 --release
```

Output: `target/wasm32-wasip2/release/wasi-nn-slm.wasm` (a component, not a
core module — verify with `wasm-tools component wit ./...wasm`).

## Get the model

```sh
./scripts/fetch-model.sh        # default: q4 variant (~323 MB)
VARIANT=fp32 ./scripts/fetch-model.sh   # ~1.14 GB, no quantisation quirks
```

Requires `python3 -m pip install onnx` for the external-data merge step.
Produces:

    models/tokenizer.json
    models/gemma3_270m.onnx

## Run

You need a `wasmtime` built with the `wasmtime-wasi-nn/onnx` feature. The
default prebuilt CLI does **not** ship it — install from source:

```sh
cargo install --git https://github.com/bytecodealliance/wasmtime \
    --features=wasi-nn,wasmtime-wasi-nn/onnx,wasmtime-wasi-nn/onnx-download \
    wasmtime-cli
```

Then:

```sh
wasmtime run \
    -S nn \
    --dir models::/models \
    target/wasm32-wasip2/release/wasi-nn-slm.wasm
```

REPL:

```
>>> hello
…
>>> /clear         # reset conversation history
>>> /bye           # exit
```

## Porting to other wasi-nn runtimes

The component imports four interfaces from `wasi:nn@0.2.0-rc-2024-10-28`:
`tensor`, `graph`, `inference`, `errors`. Any host that implements these
plus an ONNX backend should be able to load and run this binary as-is. The
guest assumes:

- `graph::load` accepts a single ONNX `graph-builder` (raw model bytes).
- The ONNX backend can drive the `gemma3_270m_*` IR with all 36 KV-cache
  inputs supplied as zero-length tensors on the first step.

If the host expects a host-named graph instead of bytes-in, swap the
`std::fs::read` + `load(...)` block in `src/main.rs` for a
`load_by_name("gemma3-270m")` call.

## Known limitations

- **Speed.** No KV cache → O(N²) tokens. A 50-token reply on a short prompt
  takes tens of seconds on CPU even with the q4 variant.
- **Logits copy.** Each step copies a `[1, seq_len, 262144]` f32 logits
  tensor across the wit ABI (~200 MB at seq_len=200). A KV-cache pass would
  reduce this to `[1, 1, 262144]` ≈ 1 MB.
- **No sampling.** Greedy argmax only — no temperature/top-p.
