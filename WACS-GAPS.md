# WACS gap report — wasi-nn backend coverage

No WACS-side gaps. The wasi-p2 WIT and wasi-p1 WITX ABIs are both
end-to-end against `WACS.WASI.NN 0.4.0` + `WACS.Cli 1.7.4`, and
new backends (e.g. `WACS.WASI.NN.OpenVino`) auto-wire into the
Preview2 DI bundle through `BuildAutoDiscoveredCallback` — no
WACS edit needed to add a backend NuGet.

## Platform note: OpenVINO on macOS arm64

The semantic-search demo (`scripts/run-embed.sh` →
`WACS.WASI.NN.OpenVino`) is gated to Linux x86_64 / arm64 and
Windows. The cause is upstream-of-WACS: Intel's
`OpenVINO.runtime.macos-arm64` NuGet stops at **2024.4.0.1** while
the OpenVINO Python release that produces IR is at **2025.x+**.
The IR-format skew trips `Core.read_model: Incorrect weights in
bin file!`. Until a newer macOS arm64 native NuGet ships, the
scripts hard-exit with a clear error on `Darwin`.

`WACS.WASI.NN.OpenVino` 0.1.2 includes a
`tools/fetch-openvino-native.sh` helper that overlays Intel's
official 2025.4.1 macOS arm64 tarball over the NuGet-staged 2024.4
dylibs — a viable manual workaround for users who want to run the
demo on macOS today. Not wired into this repo's setup so the
default install stays within NuGet-pinned native versions.

The other four backends (OnnxRuntime / OnnxRuntimeGenAI /
LlamaSharp / TorchSharp) work on macOS arm64 unchanged.
