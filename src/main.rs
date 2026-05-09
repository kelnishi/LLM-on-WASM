//! Minimal wasi-nn SLM runner targeting wasm32-wasip2.
//!
//! Loads Gemma 3 270M (ONNX) via the wasi-nn `load` host call, runs a
//! greedy-argmax generation loop without a KV cache (every step re-feeds
//! the entire prompt, so it is correct but slow), and exposes a small
//! Ollama-style stdin REPL.

use std::io::{self, BufRead, Write};

use tokenizers::Tokenizer;

mod bindings {
    wit_bindgen::generate!({
        path: "wit",
        world: "ml",
    });
}

use bindings::wasi::nn::errors::Error as NnError;
use bindings::wasi::nn::graph::{load, ExecutionTarget, Graph, GraphEncoding};
use bindings::wasi::nn::inference::GraphExecutionContext;
use bindings::wasi::nn::tensor::{Tensor, TensorType};

const MODEL_PATH: &str = "/models/gemma3_270m.onnx";
const TOKENIZER_PATH: &str = "/models/tokenizer.json";

// Gemma 3 270M architecture (from config.json).
const NUM_LAYERS: usize = 18;
const NUM_KV_HEADS: u32 = 1;
const HEAD_DIM: u32 = 256;
const VOCAB_SIZE: usize = 262_144;

// Defaults used as a fallback if tokenizer lookup fails.
const DEFAULT_BOS: u32 = 2;
const DEFAULT_EOS: u32 = 1;
const DEFAULT_END_OF_TURN: u32 = 106;

const MAX_NEW_TOKENS: usize = 256;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    eprintln!("loading tokenizer ({TOKENIZER_PATH})…");
    let tokenizer = Tokenizer::from_file(TOKENIZER_PATH)
        .map_err(|e| format!("tokenizer: {e}"))?;

    let bos = tokenizer.token_to_id("<bos>").unwrap_or(DEFAULT_BOS);
    let eos = tokenizer.token_to_id("<eos>").unwrap_or(DEFAULT_EOS);
    let eot = tokenizer
        .token_to_id("<end_of_turn>")
        .unwrap_or(DEFAULT_END_OF_TURN);

    eprintln!("loading model ({MODEL_PATH})…");
    let model_bytes = std::fs::read(MODEL_PATH)
        .map_err(|e| format!("read model: {e}"))?;
    let graph = load(&[model_bytes], GraphEncoding::Onnx, ExecutionTarget::Cpu)
        .map_err(fmt_nn_err)?;

    eprintln!(
        "ready — gemma-3-270m via wasi-nn (ONNX). type a message, /bye to exit.\n"
    );

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut transcript: Vec<(String, String)> = Vec::new();

    loop {
        write!(stdout, ">>> ")?;
        stdout.flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            writeln!(stdout)?;
            break;
        }
        let user = line.trim();
        if user.is_empty() {
            continue;
        }
        if matches!(user, "/bye" | "/exit" | "/quit") {
            break;
        }
        if user == "/clear" {
            transcript.clear();
            eprintln!("(context cleared)");
            continue;
        }

        let prompt = render_chat(&transcript, user);
        let reply = generate(&graph, &tokenizer, &prompt, bos, eos, eot)?;
        writeln!(stdout)?;
        transcript.push((user.to_string(), reply));
    }
    Ok(())
}

/// Render the Gemma 3 chat template ending with an open `<start_of_turn>model`
/// turn so the model can complete the assistant reply.
fn render_chat(history: &[(String, String)], user: &str) -> String {
    let mut s = String::from("<bos>");
    for (u, a) in history {
        s.push_str("<start_of_turn>user\n");
        s.push_str(u);
        s.push_str("<end_of_turn>\n<start_of_turn>model\n");
        s.push_str(a);
        s.push_str("<end_of_turn>\n");
    }
    s.push_str("<start_of_turn>user\n");
    s.push_str(user);
    s.push_str("<end_of_turn>\n<start_of_turn>model\n");
    s
}

fn generate(
    graph: &Graph,
    tokenizer: &Tokenizer,
    prompt: &str,
    _bos: u32,
    eos: u32,
    eot: u32,
) -> Result<String> {
    let encoding = tokenizer
        .encode(prompt, false)
        .map_err(|e| format!("encode: {e}"))?;
    let mut tokens: Vec<i64> = encoding.get_ids().iter().map(|&t| t as i64).collect();

    let ctx = graph.init_execution_context().map_err(fmt_nn_err)?;

    let mut new_tokens: Vec<u32> = Vec::with_capacity(MAX_NEW_TOKENS);
    let mut printed_bytes = 0usize;
    let mut stdout = io::stdout();

    for _ in 0..MAX_NEW_TOKENS {
        let next = step(&ctx, &tokens)?;

        if next == eos || next == eot {
            break;
        }
        new_tokens.push(next);
        tokens.push(next as i64);

        // Stream by re-decoding all generated tokens and printing only the new
        // suffix. This keeps multi-byte / multi-token UTF-8 boundaries safe.
        if let Ok(text) = tokenizer.decode(&new_tokens, true) {
            if text.len() > printed_bytes && text.is_char_boundary(printed_bytes) {
                stdout.write_all(text[printed_bytes..].as_bytes())?;
                stdout.flush()?;
                printed_bytes = text.len();
            }
        }
    }

    let full = tokenizer
        .decode(&new_tokens, true)
        .map_err(|e| format!("decode: {e}"))?;
    Ok(full)
}

/// Single forward pass: build all required ONNX inputs (no KV cache — past
/// tensors are passed in with `past_seq = 0`), call `compute`, return the
/// argmax of the last-position logits.
fn step(ctx: &GraphExecutionContext, tokens: &[i64]) -> Result<u32> {
    let seq_len = tokens.len();
    let seq_len_u32 = seq_len as u32;

    let mut inputs: Vec<(String, Tensor)> = Vec::with_capacity(3 + 2 * NUM_LAYERS);

    inputs.push((
        "input_ids".to_string(),
        Tensor::new(&[1, seq_len_u32], TensorType::I64, &i64_le_bytes(tokens)),
    ));

    let attn_mask: Vec<i64> = vec![1; seq_len];
    inputs.push((
        "attention_mask".to_string(),
        Tensor::new(&[1, seq_len_u32], TensorType::I64, &i64_le_bytes(&attn_mask)),
    ));

    let position_ids: Vec<i64> = (0..seq_len as i64).collect();
    inputs.push((
        "position_ids".to_string(),
        Tensor::new(
            &[1, seq_len_u32],
            TensorType::I64,
            &i64_le_bytes(&position_ids),
        ),
    ));

    // Empty KV cache: shape [1, num_kv_heads, 0, head_dim], so zero bytes.
    let empty_kv_dims: [u32; 4] = [1, NUM_KV_HEADS, 0, HEAD_DIM];
    for layer in 0..NUM_LAYERS {
        inputs.push((
            format!("past_key_values.{layer}.key"),
            Tensor::new(&empty_kv_dims, TensorType::Fp32, &[]),
        ));
        inputs.push((
            format!("past_key_values.{layer}.value"),
            Tensor::new(&empty_kv_dims, TensorType::Fp32, &[]),
        ));
    }

    let outputs = ctx.compute(inputs).map_err(fmt_nn_err)?;

    let logits = outputs
        .iter()
        .find(|(name, _)| name == "logits")
        .ok_or("model produced no `logits` output")?;
    let data = logits.1.data();

    // logits: [1, seq_len, VOCAB_SIZE] f32, row-major.
    let last_off = (seq_len - 1) * VOCAB_SIZE * 4;
    let last = data
        .get(last_off..last_off + VOCAB_SIZE * 4)
        .ok_or("logits buffer shorter than expected")?;

    Ok(argmax_f32_le(last))
}

fn argmax_f32_le(bytes: &[u8]) -> u32 {
    let mut best_id: u32 = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if v > best_val {
            best_val = v;
            best_id = i as u32;
        }
    }
    best_id
}

fn i64_le_bytes(v: &[i64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 8);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn fmt_nn_err(e: NnError) -> String {
    format!("wasi-nn {:?}: {}", e.code(), e.data())
}
