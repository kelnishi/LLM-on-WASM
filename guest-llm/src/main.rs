//! Minimal GGML/GGUF wasi-nn REPL — WasmEdge convention.
//!
//! Loads a GGUF by name (the host scans `WACS_WASINN_GGUF_DIR` and
//! registers each `*.gguf` under its filename stem). Each user line is
//! sent as a single `U8` tensor whose bytes are the UTF-8 prompt; the
//! host returns a single `U8` tensor whose bytes are the UTF-8 reply.
//! All tokenization, sampling, and chat templating happens host-side
//! inside LlamaSharp — the guest just shuttles bytes.

use std::io::{self, BufRead, Write};

wit_bindgen::generate!({ path: "wit", world: "ml" });

use wasi::nn::errors::Error as NnError;
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};

const MODEL_NAME: &str = "qwen2.5-0.5b-instruct-q4_k_m";

type Result<T> = std::result::Result<T, String>;

fn fmt_nn_err(e: NnError) -> String {
    format!("wasi-nn {:?}: {}", e.code(), e.data())
}

fn main() -> Result<()> {
    eprintln!("loading GGUF '{MODEL_NAME}' (host resolves via $WACS_WASINN_GGUF_DIR)…");
    let graph: Graph = load_by_name(MODEL_NAME).map_err(fmt_nn_err)?;

    eprintln!("ready — {MODEL_NAME} via wasi-nn (LlamaSharp). type a message, /bye to exit.\n");

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

        // The WasmEdge GGUF convention takes a single U8 input tensor
        // carrying the raw UTF-8 prompt bytes; LlamaSharp does its own
        // tokenization and chat-template wrapping host-side.
        let prompt_bytes = user.as_bytes();
        let prompt_len = prompt_bytes.len() as u32;
        let input = Tensor::new(&[prompt_len], TensorType::U8, prompt_bytes);

        let ctx: GraphExecutionContext =
            graph.init_execution_context().map_err(fmt_nn_err)?;
        let outputs = ctx
            .compute(vec![("0".to_string(), input)])
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
