//! Native host for the wasi-nn-slm wasm component.
//!
//! Wires wasi-cli (stdio + filesystem) and wasi-nn (ONNX backend) into a
//! `Linker`, instantiates the component as a wasi:cli/run command, and
//! transfers control to the guest. The guest reads the model + tokenizer
//! from a preopened directory, so the host's only job here is bookkeeping.

use std::env;
use std::path::PathBuf;
use std::process::ExitCode;

use wasmtime::component::{Component, Linker, ResourceTable};
use wasmtime::{Config, Engine, Store};
use wasmtime_wasi::p2::bindings::sync::Command;
use wasmtime_wasi::{DirPerms, FilePerms, WasiCtx, WasiCtxView, WasiView};
use wasmtime_wasi_nn::wit::{WasiNnCtx, WasiNnView};
use wasmtime_wasi_nn::{Backend, InMemoryRegistry};

mod backend;
use backend::OnnxBackend;

const DEFAULT_WASM: &str = "target/wasm32-wasip2/release/wasi-nn-slm.wasm";
const DEFAULT_HOST_DIR: &str = "models";
const GUEST_DIR: &str = "/models";

struct Ctx {
    wasi: WasiCtx,
    nn: WasiNnCtx,
    table: ResourceTable,
}

impl WasiView for Ctx {
    fn ctx(&mut self) -> WasiCtxView<'_> {
        WasiCtxView {
            ctx: &mut self.wasi,
            table: &mut self.table,
        }
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> wasmtime::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();

    let mut args = env::args().skip(1);
    let wasm_path: PathBuf = args
        .next()
        .unwrap_or_else(|| DEFAULT_WASM.into())
        .into();
    let host_dir: PathBuf = args
        .next()
        .unwrap_or_else(|| DEFAULT_HOST_DIR.into())
        .into();

    let engine = Engine::new(Config::new().wasm_component_model(true))?;
    let mut linker: Linker<Ctx> = Linker::new(&engine);

    wasmtime_wasi::p2::add_to_linker_sync(&mut linker)?;
    wasmtime_wasi_nn::wit::add_to_linker(&mut linker, |c: &mut Ctx| {
        WasiNnView::new(&mut c.table, &mut c.nn)
    })?;

    let mut wasi_b = WasiCtx::builder();
    wasi_b
        .inherit_stdio()
        .inherit_env()
        .preopened_dir(&host_dir, GUEST_DIR, DirPerms::READ, FilePerms::READ)?;
    let wasi = wasi_b.build();

    let backend: Backend = OnnxBackend::default().into();
    let registry = InMemoryRegistry::new().into();
    let nn = WasiNnCtx::new([backend], registry);

    let ctx = Ctx {
        wasi,
        nn,
        table: ResourceTable::new(),
    };

    let component = Component::from_file(&engine, &wasm_path)?;
    let mut store = Store::new(&engine, ctx);
    // The default hostcall fuel limit caps host<->guest copies at a few MB.
    // We pass a 300MB+ ONNX model on load and a ~200MB logits tensor per
    // generation step, so disable the limit.
    store.set_hostcall_fuel(usize::MAX);
    let cmd = Command::instantiate(&mut store, &component, &linker)?;
    cmd.wasi_cli_run()
        .call_run(&mut store)?
        .map_err(|()| wasmtime::Error::msg("guest exited non-zero"))
}
