//! Semantic-search demo against OpenVINO via wasi-nn.
//!
//! Loads `all-MiniLM-L6-v2` as an OpenVINO IR pair (XML + weights bin)
//! through the wasi-nn `load` host call with `GraphEncoding::Openvino`,
//! tokenizes a small corpus of reference sentences at startup, then
//! ranks each typed query against the corpus by cosine similarity over
//! mean-pooled token embeddings.
//!
//! Input shape is fixed at `[1, 64]` (set at the IR conversion step in
//! `scripts/fetch-embed-model.sh`) — wasi-nn requires concrete shapes,
//! so longer inputs get truncated and shorter ones zero-padded with a
//! masked attention.

use std::io::{self, BufRead, Write};

use tokenizers::Tokenizer;

mod bindings {
    wit_bindgen::generate!({
        path: "wit",
        world: "ml",
    });
}

use bindings::wasi::nn::errors::Error as NnError;
use bindings::wasi::nn::graph::{load, ExecutionTarget, GraphEncoding};
use bindings::wasi::nn::inference::GraphExecutionContext;
use bindings::wasi::nn::tensor::{Tensor, TensorType};

const IR_XML_PATH:   &str = "/models/minilm.xml";
const IR_BIN_PATH:   &str = "/models/minilm.bin";
const TOKENIZER:     &str = "/models/sentence-tokenizer.json";

// Match the seq_len pinned at IR conversion time. The host reshaped
// every input to [1, SEQ_LEN] before compile_model — the guest must
// pad/truncate to match or OpenVINO rejects the input.
const SEQ_LEN: usize = 64;
const EMBED_DIM: usize = 384;

// A small mixed-topic corpus the user can search through. Picked to
// give the demo something to rank — a query like "lunar landing"
// should rank "The Apollo 11 mission..." above "Photosynthesis...".
const CORPUS: &[&str] = &[
    "The Apollo 11 mission landed humans on the Moon in 1969.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "WebAssembly is a portable compilation target for high-level languages.",
    "Espresso is a concentrated form of coffee brewed under pressure.",
    "The capital of France is Paris, on the river Seine.",
    "Quicksort is a divide-and-conquer sorting algorithm.",
    "Saturn's rings are made mostly of ice and rock fragments.",
    "Tokyo is the most populous metropolitan area in the world.",
    "A neural network is a function approximator built from simple linear units.",
    "The Pacific is the largest and deepest of Earth's oceans.",
];

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> Result<()> {
    eprintln!("loading tokenizer ({TOKENIZER})…");
    let tokenizer = Tokenizer::from_file(TOKENIZER)
        .map_err(|e| format!("tokenizer: {e}"))?;

    eprintln!("loading OpenVINO IR ({IR_XML_PATH} + .bin)…");
    let xml = std::fs::read(IR_XML_PATH).map_err(|e| format!("read xml: {e}"))?;
    let bin = std::fs::read(IR_BIN_PATH).map_err(|e| format!("read bin: {e}"))?;
    let graph = load(&[xml, bin], GraphEncoding::Openvino, ExecutionTarget::Cpu)
        .map_err(fmt_nn_err)?;
    let ctx = graph.init_execution_context().map_err(fmt_nn_err)?;

    eprintln!("embedding {} reference sentences…", CORPUS.len());
    let corpus_vecs: Vec<[f32; EMBED_DIM]> = CORPUS
        .iter()
        .map(|s| embed(&ctx, &tokenizer, s))
        .collect::<Result<Vec<_>>>()?;

    eprintln!("ready — type a query, /bye to exit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        write!(stdout, ">>> ")?;
        stdout.flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            writeln!(stdout)?;
            break;
        }
        let query = line.trim();
        if query.is_empty() {
            continue;
        }
        if matches!(query, "/bye" | "/exit" | "/quit") {
            break;
        }

        let q_vec = embed(&ctx, &tokenizer, query)?;
        let mut scored: Vec<(usize, f32)> = corpus_vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine(&q_vec, v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        writeln!(stdout)?;
        for (rank, (i, score)) in scored.iter().take(3).enumerate() {
            writeln!(stdout, "  {}. [{:.3}] {}", rank + 1, score, CORPUS[*i])?;
        }
        writeln!(stdout)?;
    }
    Ok(())
}

fn embed(
    ctx: &GraphExecutionContext,
    tokenizer: &Tokenizer,
    text: &str,
) -> Result<[f32; EMBED_DIM]> {
    let encoding = tokenizer
        .encode(text, true)
        .map_err(|e| format!("encode: {e}"))?;
    let raw_ids = encoding.get_ids();
    let raw_mask = encoding.get_attention_mask();
    let raw_types = encoding.get_type_ids();

    // Truncate / zero-pad to SEQ_LEN — IR was compiled with a fixed
    // shape, so input tensors must match exactly.
    let mut input_ids = [0i64; SEQ_LEN];
    let mut attention_mask = [0i64; SEQ_LEN];
    let mut token_type_ids = [0i64; SEQ_LEN];
    let n = raw_ids.len().min(SEQ_LEN);
    for i in 0..n {
        input_ids[i] = raw_ids[i] as i64;
        attention_mask[i] = raw_mask[i] as i64;
        token_type_ids[i] = raw_types[i] as i64;
    }

    let dims = [1u32, SEQ_LEN as u32];
    let inputs = vec![
        ("input_ids".to_string(),
            Tensor::new(&dims, TensorType::I64, &i64_le_bytes(&input_ids))),
        ("attention_mask".to_string(),
            Tensor::new(&dims, TensorType::I64, &i64_le_bytes(&attention_mask))),
        ("token_type_ids".to_string(),
            Tensor::new(&dims, TensorType::I64, &i64_le_bytes(&token_type_ids))),
    ];

    let outputs = ctx.compute(inputs).map_err(fmt_nn_err)?;
    // OpenVINO model has a single output node — last_hidden_state
    // shaped [1, SEQ_LEN, EMBED_DIM] FP32. Mean-pool over the
    // attention-masked positions to get a sentence embedding.
    let last_hidden = &outputs
        .first()
        .ok_or("model produced no outputs")?
        .1;
    let data = last_hidden.data();
    if data.len() != SEQ_LEN * EMBED_DIM * 4 {
        return Err(format!(
            "unexpected last_hidden_state size: {} (want {})",
            data.len(),
            SEQ_LEN * EMBED_DIM * 4
        )
        .into());
    }

    let mut pooled = [0.0f32; EMBED_DIM];
    let mut mask_sum = 0.0f32;
    for t in 0..SEQ_LEN {
        let m = attention_mask[t] as f32;
        if m == 0.0 {
            continue;
        }
        mask_sum += m;
        let row_off = t * EMBED_DIM * 4;
        for d in 0..EMBED_DIM {
            let off = row_off + d * 4;
            let v = f32::from_le_bytes([
                data[off], data[off + 1], data[off + 2], data[off + 3],
            ]);
            pooled[d] += v;
        }
    }
    if mask_sum > 0.0 {
        for d in 0..EMBED_DIM {
            pooled[d] /= mask_sum;
        }
    }

    // L2-normalize so cosine similarity is just a dot product.
    let norm = pooled.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for d in 0..EMBED_DIM {
        pooled[d] /= norm;
    }
    Ok(pooled)
}

fn cosine(a: &[f32; EMBED_DIM], b: &[f32; EMBED_DIM]) -> f32 {
    let mut dot = 0.0f32;
    for d in 0..EMBED_DIM {
        dot += a[d] * b[d];
    }
    dot
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
