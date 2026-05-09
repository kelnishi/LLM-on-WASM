# wasi-nn-slm

A minimal stub that runs **Gemma 3 270M (ONNX)** inside a `wasm32-wasip2`
component, using **wasi-nn** for inference and the HuggingFace `tokenizers`
crate (with the `unstable_wasm` feature) for tokenization. Exposes a small
Ollama-style stdin REPL.

The point is portability: the same `.wasm` should run under any wasi-p2
runtime that implements wasi-nn's WIT interface with an ONNX backend.

## Layout

This is a Cargo workspace with two crates:

    Cargo.toml                  workspace; default-members = ["host"]
    rust-toolchain.toml         pins wasm32-wasip2 + native stable
    scripts/fetch-model.sh      pulls model + tokenizer, inlines external weights
    guest/                      the wasi-p2 component (the actual portable artefact)
        Cargo.toml              wit-bindgen + tokenizers (unstable_wasm)
        wit/wasi-nn.wit         vendored from WebAssembly/wasi-nn (0.2.0-rc-2024-10-28)
        src/main.rs             REPL, chat-template, no-cache greedy generation
    host/                       a tiny native runner — only needed during dev
        Cargo.toml              wasmtime 44 + custom ort-backed wasi-nn backend
        src/main.rs             linker setup, preopens, instantiates Command
        src/backend.rs          custom OnnxBackend impl (see below for why)

## The host runner

`wasmtime`'s prebuilt CLI doesn't ship the wasi-nn ONNX backend, and even
when you build it from source with `wasmtime-wasi-nn/onnx`, the bundled
backend only handles **`Fp32` tensors** — it `unimplemented!()`s on
anything else, which is unusable for any LLM (input_ids and attention_mask
are int64).

`host/src/backend.rs` therefore implements the public `BackendInner` /
`BackendGraph` / `BackendExecutionContext` traits ourselves on top of
`ort` directly, supporting `I64`, `Fp32`, `Fp16`, `I32`, and `U8` for both
inputs and outputs. We pass it to `WasiNnCtx::new` instead of the bundled
`OnnxBackend`. ~150 LOC.

A second wrinkle: `ort`'s `Tensor::from_array((Vec<i64>, Vec<T>))` rejects
any dimension < 1, which makes empty KV-cache tensors (`[1, 1, 0, 256]`)
unconstructable. The backend goes through `ndarray::ArrayD::from_shape_vec`
instead, which has no such check.

## Build and run

One-time setup (~1 GB download for the fp32 model):

```sh
python3 -m pip install onnx     # for the external-data inlining step
./scripts/fetch-model.sh        # default VARIANT=fp32 (~1.14 GB)
```

Build the guest component (release recommended — debug is ~50 MB and slow):

```sh
cargo build -p wasi-nn-slm --target wasm32-wasip2 --release
```

Build and run the host:

```sh
cargo run -p wasi-nn-slm-host --release
```

The host defaults to loading `target/wasm32-wasip2/release/wasi-nn-slm.wasm`
and preopening `./models` as `/models` inside the guest.

REPL:

    >>> hello
    Hello! How can I help you today?
    >>> /clear         # reset conversation history
    >>> /bye           # exit (or Ctrl-D)

## Architecture notes

- **No KV cache.** Every step retokenises and re-feeds the entire prompt
  with empty `past_key_values.{i}.key/value` tensors (shape `[1, 1, 0, 256]`,
  zero bytes). Quadratic and slow but keeps the stub small. `present.*`
  outputs are ignored.
- **External data is inlined at fetch time.** The wasi-nn `graph::load`
  ABI takes raw bytes; ONNX `external_data` references would never resolve
  from the sandbox, so `fetch-model.sh` calls
  `onnx.save(..., save_as_external_data=False)` to merge the weights into
  a single self-contained `gemma3_270m.onnx`.
- **Chat template** is hand-rolled rather than running Jinja in wasm —
  the Gemma format is small enough to fit in `render_chat()`.
- **Streaming** is done by re-decoding all generated tokens after each
  step and printing the byte-suffix at safe UTF-8 boundaries.
- **Hostcall fuel** is lifted to `usize::MAX` on the wasmtime store —
  default cap of a few MB blocks the 1+ GB model load and ~200 MB
  per-step logits copy.

## Porting to other wasi-nn runtimes

The component imports four interfaces from `wasi:nn@0.2.0-rc-2024-10-28`:
`tensor`, `graph`, `inference`, `errors`. Any host that implements these
plus an ONNX backend with `I64` input support should be able to load
and run this binary as-is. The guest assumes:

- `graph::load` accepts a single ONNX `graph-builder` (raw model bytes,
  external data already inlined).
- The ONNX backend can drive the `gemma3_270m.onnx` IR with all 36
  KV-cache inputs supplied as zero-length `[1, 1, 0, 256]` Fp32 tensors
  on every step.

If your target host expects a host-named graph instead of bytes-in,
swap the `std::fs::read` + `load(...)` block in `guest/src/main.rs` for
a `load_by_name("gemma3-270m")` call.

## Known limitations

- **Speed.** No KV cache → O(N²) per step. A 50-token reply on a short
  prompt takes tens of seconds on CPU.
- **Logits copy.** Each step copies a `[1, seq_len, 262144]` f32 logits
  tensor across the wit ABI (~200 MB at seq_len=200). A KV-cache pass
  would shrink this to `[1, 1, 262144]` ≈ 1 MB.
- **No sampling.** Greedy argmax only — no temperature/top-p.
- **`q4` / `q4f16` model variants don't work** with `ort` 2.0.0-rc.10
  because they use `com.microsoft.GatherBlockQuantized` with a `bits`
  attribute the bundled ORT doesn't recognise. fp32 is the default for
  this reason.
