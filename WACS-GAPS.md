# WACS gap report — wasi-nn backend coverage

No open gaps. All five backends — `OnnxRuntime`, `OnnxRuntimeGenAI`,
`LlamaSharp`, `TorchSharp`, `OpenVino` — run end-to-end against
the published NuGet stack:

- `WACS.Cli` 1.7.6
- `WACS.WASI.NN` 0.4.0
- `WACS.WASI.NN.OpenVino` 0.2.1 (others at the versions pinned in
  `tools/Backends/Backends.csproj`)
- `OpenVINO.runtime.macos-arm64` 2026.1.0 / `OpenVINO.runtime.win`
  2026.0.0 / `OpenVINO.runtime.ubuntu.{22-x86_64,20-arm64}`
  2024.4.0.1

Verified on macOS arm64 (`Darwin 25.4.0`) with the
`scripts/run-embed.sh` semantic-search demo:

```
>>> lunar landing
  1. [0.592] The Apollo 11 mission landed humans on the Moon in 1969.

>>> where is the city of paris
  1. [0.750] The capital of France is Paris, on the river Seine.
```

`BuildAutoDiscoveredCallback` wires each `IWasiNNBackendRegistration`
into the Preview2 DI bundle without per-backend edits to
`WasiPreview2RuntimeScope`, so adding a new backend NuGet to
`tools/Backends/Backends.csproj` is the only step needed on this
side.
