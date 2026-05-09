//! Custom wasi-nn ONNX backend that supports Int64/Fp32/Fp16/Int32/Uint8
//! tensors via `ort` directly, replacing wasmtime-wasi-nn's bundled backend
//! (which only handles Fp32).

use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use ort::session::{builder::GraphOptimizationLevel, Session, SessionInputValue};
use ort::tensor::TensorElementType;
use ort::value::{Tensor as OrtTensor, ValueType};
use wasmtime_wasi_nn::backend::{
    BackendError, BackendExecutionContext, BackendFromDir, BackendGraph, BackendInner, Id,
    NamedTensor,
};
use wasmtime_wasi_nn::wit::types::{ExecutionTarget, GraphEncoding, Tensor, TensorType};
use wasmtime_wasi_nn::{ExecutionContext, Graph};

#[derive(Default)]
pub struct OnnxBackend;

impl BackendInner for OnnxBackend {
    fn encoding(&self) -> GraphEncoding {
        GraphEncoding::Onnx
    }

    fn load(
        &mut self,
        builders: &[&[u8]],
        _target: ExecutionTarget,
    ) -> Result<Graph, BackendError> {
        if builders.len() != 1 {
            return Err(BackendError::InvalidNumberOfBuilders(1, builders.len()));
        }
        let session = Session::builder()
            .map_err(ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(ort_err)?
            .commit_from_memory(builders[0])
            .map_err(ort_err)?;

        let g: Box<dyn BackendGraph> = Box::new(OnnxGraph(Arc::new(Mutex::new(session))));
        Ok(g.into())
    }

    fn as_dir_loadable<'a>(&'a mut self) -> Option<&'a mut dyn BackendFromDir> {
        None
    }
}

struct OnnxGraph(Arc<Mutex<Session>>);

impl BackendGraph for OnnxGraph {
    fn init_execution_context(&self) -> Result<ExecutionContext, BackendError> {
        let ctx: Box<dyn BackendExecutionContext> = Box::new(OnnxContext(self.0.clone()));
        Ok(ctx.into())
    }
}

struct OnnxContext(Arc<Mutex<Session>>);

impl BackendExecutionContext for OnnxContext {
    fn set_input(&mut self, _id: Id, _t: &Tensor) -> Result<(), BackendError> {
        Err(backend_msg("witx set_input not supported (use the wit compute path)"))
    }

    fn get_output(&mut self, _id: Id) -> Result<Tensor, BackendError> {
        Err(backend_msg("witx get_output not supported (use the wit compute path)"))
    }

    fn compute(
        &mut self,
        inputs: Option<Vec<NamedTensor>>,
    ) -> Result<Option<Vec<NamedTensor>>, BackendError> {
        let inputs = inputs.ok_or_else(|| backend_msg("compute called without wit inputs"))?;

        let mut session_inputs: Vec<(Cow<'static, str>, SessionInputValue<'static>)> =
            Vec::with_capacity(inputs.len());
        for nt in inputs {
            let v = build_input(&nt)?;
            session_inputs.push((Cow::Owned(nt.name), v));
        }

        let mut session = self.0.lock().unwrap();
        let outputs = session.run(session_inputs).map_err(ort_err)?;

        let mut out = Vec::with_capacity(outputs.len());
        for (name, val) in outputs.iter() {
            let (dims, ty, data) = extract_output(&val)?;
            out.push(NamedTensor {
                name: name.to_string(),
                tensor: Tensor::new(dims, ty, data),
            });
        }
        Ok(Some(out))
    }
}

fn build_input(nt: &NamedTensor) -> Result<SessionInputValue<'static>, BackendError> {
    // Use ndarray's `ArrayD::from_shape_vec` rather than the `(Vec<i64>, Vec<T>)`
    // overload of `Tensor::from_array`, because the latter enforces every
    // dimension >= 1, which makes empty KV-cache tensors (`[1, 1, 0, 256]`)
    // unusable.
    let dims_usize: Vec<usize> = nt.tensor.dimensions.iter().map(|&d| d as usize).collect();
    let shape = ndarray::IxDyn(&dims_usize);
    let bytes = &nt.tensor.data;

    let v: SessionInputValue<'static> = match nt.tensor.ty {
        TensorType::Fp32 => {
            let buf: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let arr = ndarray::ArrayD::from_shape_vec(shape, buf).map_err(shape_err)?;
            OrtTensor::from_array(arr).map_err(ort_err)?.into()
        }
        TensorType::Fp16 => {
            let buf: Vec<half::f16> = bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
                .collect();
            let arr = ndarray::ArrayD::from_shape_vec(shape, buf).map_err(shape_err)?;
            OrtTensor::from_array(arr).map_err(ort_err)?.into()
        }
        TensorType::I64 => {
            let buf: Vec<i64> = bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            let arr = ndarray::ArrayD::from_shape_vec(shape, buf).map_err(shape_err)?;
            OrtTensor::from_array(arr).map_err(ort_err)?.into()
        }
        TensorType::I32 => {
            let buf: Vec<i32> = bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let arr = ndarray::ArrayD::from_shape_vec(shape, buf).map_err(shape_err)?;
            OrtTensor::from_array(arr).map_err(ort_err)?.into()
        }
        TensorType::U8 => {
            let arr = ndarray::ArrayD::from_shape_vec(shape, bytes.clone())
                .map_err(shape_err)?;
            OrtTensor::from_array(arr).map_err(ort_err)?.into()
        }
        other => {
            return Err(BackendError::UnsupportedTensorType(format!("{other:?}")));
        }
    };
    Ok(v)
}

fn shape_err(e: ndarray::ShapeError) -> BackendError {
    BackendError::BackendAccess(wasmtime::Error::msg(format!("shape: {e}")))
}

fn extract_output(val: &ort::value::ValueRef<'_>) -> Result<(Vec<u32>, TensorType, Vec<u8>), BackendError> {
    let (elem_ty, dims) = match val.dtype() {
        ValueType::Tensor { ty, shape, .. } => {
            let dims: Vec<u32> = shape.iter().map(|&d| d as u32).collect();
            (*ty, dims)
        }
        _ => return Err(backend_msg("non-tensor outputs are not supported")),
    };

    let (ty, data): (TensorType, Vec<u8>) = match elem_ty {
        TensorElementType::Float32 => {
            let (_, slice) = val.try_extract_tensor::<f32>().map_err(ort_err)?;
            (TensorType::Fp32, slice.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
        TensorElementType::Float16 => {
            let (_, slice) = val.try_extract_tensor::<half::f16>().map_err(ort_err)?;
            (TensorType::Fp16, slice.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
        TensorElementType::Int64 => {
            let (_, slice) = val.try_extract_tensor::<i64>().map_err(ort_err)?;
            (TensorType::I64, slice.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
        TensorElementType::Int32 => {
            let (_, slice) = val.try_extract_tensor::<i32>().map_err(ort_err)?;
            (TensorType::I32, slice.iter().flat_map(|x| x.to_le_bytes()).collect())
        }
        TensorElementType::Uint8 => {
            let (_, slice) = val.try_extract_tensor::<u8>().map_err(ort_err)?;
            (TensorType::U8, slice.to_vec())
        }
        other => {
            return Err(BackendError::UnsupportedTensorType(format!("{other:?}")));
        }
    };
    Ok((dims, ty, data))
}

fn ort_err(e: ort::Error) -> BackendError {
    // Inline the ORT message so it survives `wasmtime::Error::to_string()` —
    // anyhow's default Display only shows the top-level error, not sources.
    BackendError::BackendAccess(wasmtime::Error::msg(format!("ort: {e}")))
}

fn backend_msg(m: &str) -> BackendError {
    BackendError::BackendAccess(wasmtime::Error::msg(m.to_string()))
}
