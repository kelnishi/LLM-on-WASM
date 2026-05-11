//! Minimal load-by-name wasi-nn REPL — WasmEdge convention.
//!
//! Backend-agnostic: any wasi-nn host implementing `graph.load-by-name`
//! with the WasmEdge U8-in / U8-out tensor convention works. Tested
//! backends:
//!
//!   - `Wacs.WASI.NN.LlamaSharp` — GGUF via llama.cpp.
//!     Default `MODEL_NAME = "qwen2.5-0.5b-instruct-q4_k_m"`; host scans
//!     `$WACS_WASINN_GGUF_DIR` for `*.gguf` files.
//!   - `Wacs.WASI.NN.OnnxRuntimeGenAI` — ONNX via Microsoft.ML.OnnxRuntimeGenAI.
//!     Set `MODEL_NAME=<onnx-genai-dir-stem>` env var; host scans
//!     `$WACS_WASINN_ONNXGENAI_DIR` for genai-config-bearing directories.
//!
//! Each user line is sent as a single `U8` tensor named `"0"` whose bytes
//! are the UTF-8 prompt; the host returns a single `U8` tensor named `"0"`
//! whose bytes are the UTF-8 reply. All tokenization, sampling, and chat
//! templating happens host-side — the guest just shuttles bytes.

use std::io::{self, BufRead, Write};

wit_bindgen::generate!({ path: "wit", world: "ml" });

use wasi::nn::errors::Error as NnError;
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};

// Default model name targets the LlamaSharp / GGUF path. Override at
// run-time via `MODEL_NAME=<stem>` env var to point at any other
// load-by-name backend (e.g., OnnxRuntimeGenAI).
const DEFAULT_MODEL_NAME: &str = "qwen2.5-0.5b-instruct-q4_k_m";

type Result<T> = std::result::Result<T, String>;

fn fmt_nn_err(e: NnError) -> String {
    format!("wasi-nn {:?}: {}", e.code(), e.data())
}

fn main() -> Result<()> {
    let model_name = std::env::var("MODEL_NAME")
        .unwrap_or_else(|_| DEFAULT_MODEL_NAME.to_string());
    eprintln!("loading model '{model_name}' via wasi-nn graph.load-by-name…");
    let graph: Graph = load_by_name(&model_name).map_err(fmt_nn_err)?;

    eprintln!("ready — {model_name} via wasi-nn (load-by-name). type a message, /bye to exit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        write!(stdout, ">>> ").map_err(|e| e.to_string())?;
        stdout.flush().map_err(|e| e.to_string())?;

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

        // Single U8 input tensor named "prompt" with the raw UTF-8
        // prompt bytes. The backend handles tokenization, chat
        // templating, sampling, and KV-cached decode host-side.
        //   - LlamaSharp ignores input names (uses inputs[0])
        //   - OnnxRuntimeGenAI dispatches by the first input name:
        //     "prompt" picks the bytes-in / bytes-out generation path
        let prompt_bytes = user.as_bytes();
        let prompt_len = prompt_bytes.len() as u32;
        let input = Tensor::new(&[prompt_len], TensorType::U8, prompt_bytes);

        let ctx: GraphExecutionContext =
            graph.init_execution_context().map_err(fmt_nn_err)?;
        let outputs = ctx
            .compute(vec![("prompt".to_string(), input)])
            .map_err(fmt_nn_err)?;

        // Single named output "0" per the WasmEdge convention; we
        // accept either "0" or the first output as a courtesy.
        let reply_tensor = outputs
            .iter()
            .find(|(name, _)| name == "0")
            .map(|(_, t)| t)
            .or_else(|| outputs.first().map(|(_, t)| t))
            .ok_or_else(|| "model produced no outputs".to_string())?;
        let reply_bytes = reply_tensor.data();
        let reply = String::from_utf8_lossy(&reply_bytes);

        writeln!(stdout, "{}", reply).map_err(|e| e.to_string())?;
        stdout.flush().ok();
    }
    Ok(())
}
