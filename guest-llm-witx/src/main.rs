//! Minimal WITX (Preview 1) wasi-nn REPL — runs on WasmEdge with the
//! GGML plugin or any other `wasi_ephemeral_nn` host.
//!
//! WasmEdge GGUF convention (same wire shape as our wasm32-wasip2
//! `guest-llm`, just over the legacy WITX ABI): the host pre-registers
//! a model under a name via `--nn-preload <name>:GGML:AUTO:<path>`,
//! the guest calls `load_by_name(name)`, feeds the prompt as a U8
//! tensor on input slot 0, calls `compute()`, reads the reply bytes
//! back from output slot 0.
//!
//! Run:
//!     wasmedge --nn-preload default:GGML:AUTO:models/qwen....gguf \
//!         target/wasm32-wasip1/release/wasi-nn-llm-witx.wasm
//!
//! Or via scripts/run-llm-wasmedge.sh which wires this up with the
//! Qwen2.5 0.5B GGUF that scripts/fetch-gguf.sh stages.

use std::io::{self, BufRead, Write};

const MODEL_NAME_DEFAULT: &str = "default";

fn main() -> Result<(), String> {
    // Host pre-registered the model under this name via --nn-preload.
    // Override at run-time with MODEL_NAME=<other> if the host
    // registered something other than "default".
    let model_name = std::env::var("MODEL_NAME")
        .unwrap_or_else(|_| MODEL_NAME_DEFAULT.to_string());

    eprintln!("loading model '{model_name}' via wasi-nn (WITX / GGML)…");
    let graph = wasi_nn::GraphBuilder::new(
        wasi_nn::GraphEncoding::Autodetec,
        wasi_nn::ExecutionTarget::AUTO,
    )
    .build_from_cache(&model_name)
    .map_err(|e| format!("load_by_name: {e:?}"))?;

    eprintln!("ready — type a message, /bye to exit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        write!(stdout, ">>> ").map_err(|e| e.to_string())?;
        stdout.flush().ok();

        let mut line = String::new();
        if stdin.lock().read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            writeln!(stdout).ok();
            break;
        }
        let user = line.trim();
        if user.is_empty() {
            continue;
        }
        if matches!(user, "/bye" | "/exit" | "/quit") {
            break;
        }

        // Fresh context per turn — keeps the example simple (each turn
        // is single-shot; no multi-turn KV continuity).
        let mut ctx = graph
            .init_execution_context()
            .map_err(|e| format!("init_execution_context: {e:?}"))?;

        // WITX wasi-nn is stateful per-context: set_input → compute →
        // get_output. (The wasi-p2 wit ABI rolls those three into a
        // single `compute(inputs) -> outputs` call.)
        let prompt = user.as_bytes();
        ctx.set_input(
            0,
            wasi_nn::TensorType::U8,
            &[prompt.len()],
            prompt,
        )
        .map_err(|e| format!("set_input: {e:?}"))?;

        ctx.compute()
            .map_err(|e| format!("compute: {e:?}"))?;

        // Pull until we either get a short read or fill the buffer.
        // The WasmEdge plugin returns the full reply on get_output(0).
        let mut buf = vec![0u8; 4 * 1024];
        let read = ctx
            .get_output(0, &mut buf)
            .map_err(|e| format!("get_output: {e:?}"))?;

        let reply = String::from_utf8_lossy(&buf[..read]);
        writeln!(stdout, "{reply}").map_err(|e| e.to_string())?;
        stdout.flush().ok();
    }

    Ok(())
}
