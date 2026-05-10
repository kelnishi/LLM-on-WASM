//! Minimal wasi-nn TorchScript guest — exercises an XOR MLP through
//! WACS's TorchSharp backend.
//!
//! Loads `xor-mlp` by name (`WACS_WASINN_TORCH_DIR/xor-mlp.pt`) and
//! evaluates the four 2-bit truth-table inputs:
//!     (0,0) -> 0   (0,1) -> 1   (1,0) -> 1   (1,1) -> 0
//! Prints the model's FP32 sigmoid output for each, plus a thresholded
//! 0/1 verdict, plus the expected truth-table value, plus a final
//! pass/fail summary.
//!
//! The TorchSharp backend follows the WasmEdge convention for input /
//! output naming: positional indices as decimal strings ("0", "1", …),
//! both inbound and outbound. This module is single-input / single-
//! output, so we use just "0".

wit_bindgen::generate!({ path: "wit", world: "ml" });

use wasi::nn::errors::Error as NnError;
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::tensor::{Tensor, TensorType};

const MODEL_NAME: &str = "xor-mlp";

type Result<T> = std::result::Result<T, String>;

fn fmt_nn_err(e: NnError) -> String {
    format!("wasi-nn {:?}: {}", e.code(), e.data())
}

fn f32_le_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn read_f32_le(bytes: &[u8]) -> f32 {
    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn evaluate(graph: &Graph, a: f32, b: f32) -> Result<f32> {
    let input_bytes = f32_le_bytes(&[a, b]);
    // Shape [1, 2]: batch=1, two-feature input vector.
    let input = Tensor::new(&[1u32, 2u32], TensorType::Fp32, &input_bytes);

    let ctx = graph.init_execution_context().map_err(fmt_nn_err)?;
    let outputs = ctx
        .compute(vec![("0".to_string(), input)])
        .map_err(fmt_nn_err)?;

    let out_tensor = outputs
        .iter()
        .find(|(name, _)| name == "0")
        .map(|(_, t)| t)
        .or_else(|| outputs.first().map(|(_, t)| t))
        .ok_or_else(|| "model produced no outputs".to_string())?;

    let data = out_tensor.data();
    if data.len() < 4 {
        return Err(format!(
            "expected at least 4 bytes (one f32) in output tensor, got {}",
            data.len()
        ));
    }
    Ok(read_f32_le(&data[..4]))
}

fn main() -> Result<()> {
    eprintln!("loading TorchScript '{MODEL_NAME}' (host resolves via $WACS_WASINN_TORCH_DIR)…");
    let graph: Graph = load_by_name(MODEL_NAME).map_err(fmt_nn_err)?;
    eprintln!("ready — {MODEL_NAME} via wasi-nn (TorchSharp / libtorch).\n");

    let cases: [(f32, f32, u8); 4] =
        [(0.0, 0.0, 0), (0.0, 1.0, 1), (1.0, 0.0, 1), (1.0, 1.0, 0)];
    let mut all_pass = true;

    for (a, b, expected) in cases {
        let p = evaluate(&graph, a, b)?;
        let pred: u8 = if p >= 0.5 { 1 } else { 0 };
        let ok = pred == expected;
        if !ok {
            all_pass = false;
        }
        println!(
            "  XOR({}, {}) -> sigmoid={:.4}  pred={}  expected={}  {}",
            a as u8,
            b as u8,
            p,
            pred,
            expected,
            if ok { "OK" } else { "FAIL" }
        );
    }

    println!("\n{}", if all_pass { "all cases pass" } else { "some cases failed" });
    if all_pass {
        Ok(())
    } else {
        Err("XOR MLP did not match the truth table".to_string())
    }
}
