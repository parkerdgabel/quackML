use std::any::Any;

use anyhow::anyhow;
use anyhow::Result;
use duckdb::params;
use std::fmt::Debug;

use crate::{
    context,
    orm::{Dataset, Hyperparams},
};
#[cfg(feature = "python")]
use pyo3::{pyfunction, PyResult, Python};

#[cfg(feature = "python")]
#[pyfunction]
pub fn r_insert_logs(project_id: i64, model_id: i64, logs: String) -> PyResult<String> {
    let result: Result<i64, _> = context::run(|conn| {
        let id_value: i64 = conn.query_row(
            "INSERT INTO quackml.logs (project_id, model_id, logs) VALUES ($1, $2, $3::JSONB) RETURNING id;",
            params![project_id, model_id, logs],
            |row| row.get(0),
        ).expect("Expected id to return");
        Ok(id_value)
    });
    Ok(format!("Inserted logs with id: {}", result.unwrap()))
}

#[cfg(feature = "python")]
#[pyfunction]
pub fn r_log(level: String, message: String) -> PyResult<String> {
    match level.as_str() {
        "info" => log::info!("{}", message),
        "warning" => log::warn!("{}", message),
        "debug" => log::debug!("{}", message),
        "error" => log::error!("{}", message),
        _ => log::info!("{}", message),
    };
    Ok(message)
}

#[cfg(feature = "python")]
#[macro_export]
macro_rules! create_pymodule {
    ($pyfile:literal) => {
        pub static PY_MODULE: once_cell::sync::Lazy<
            anyhow::Result<pyo3::Py<pyo3::types::PyModule>>,
        > = once_cell::sync::Lazy::new(|| {
            pyo3::Python::with_gil(|py| -> anyhow::Result<pyo3::Py<pyo3::types::PyModule>> {
                use $crate::bindings::TracebackError;
                let src = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), $pyfile));
                let module =
                    pyo3::types::PyModule::from_code(py, src, "transformers.py", "__main__")
                        .format_traceback(py)?;
                module.add_function(wrap_pyfunction!($crate::bindings::r_insert_logs, module)?)?;
                module.add_function(wrap_pyfunction!($crate::bindings::r_log, module)?)?;
                Ok(module.into())
            })
        });
    };
}

#[cfg(feature = "python")]
#[macro_export]
macro_rules! get_module {
    ($module:ident) => {
        match $module.as_ref() {
            Ok(module) => module,
            Err(e) => anyhow::bail!(e),
        }
    };
}

pub mod langchain;
pub mod lightgbm;
pub mod linfa;
#[cfg(feature = "python")]
pub mod python;
pub mod sklearn;
pub mod transformers;
pub mod xgboost;

pub type Fit = fn(dataset: &Dataset, hyperparams: &Hyperparams) -> Result<Box<dyn Bindings>>;

pub trait AToAny: 'static {
    fn as_any(&self) -> &dyn Any;
}

impl<T: 'static> AToAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
/// The Bindings trait that has to be implemented by all algorithm
/// providers we use in PostgresML. We don't rely on Serde serialization,
/// since scikit-learn estimators were originally serialized in pure Python as
/// pickled objects, and neither xgboost nor linfa estimators completely
/// implement serde.
pub trait Bindings: Send + Sync + Debug + AToAny {
    /// Predict a set of datapoints.
    fn predict(
        &self,
        features: &[f32],
        num_features: usize,
        num_classes: usize,
    ) -> Result<Vec<f32>>;

    /// Predict the probability of each class.
    fn predict_proba(&self, features: &[f32], num_features: usize) -> Result<Vec<f32>>;

    /// Serialize self to bytes
    fn to_bytes(&self) -> Result<Vec<u8>>;

    /// Deserialize self from bytes, with additional context
    fn from_bytes(bytes: &[u8]) -> Result<Box<dyn Bindings>>
    where
        Self: Sized;
}

#[cfg(feature = "python")]
pub trait TracebackError<T> {
    fn format_traceback(self, py: Python<'_>) -> Result<T>;
}

#[cfg(feature = "python")]
impl<T> TracebackError<T> for PyResult<T> {
    fn format_traceback(self, py: Python<'_>) -> Result<T> {
        self.map_err(|e| match e.traceback(py) {
            Some(traceback) => match traceback.format() {
                Ok(traceback) => anyhow!("{traceback} {e}"),
                Err(format_e) => anyhow!("{e}: {format_e}"),
            },
            None => anyhow!("{e}"),
        })
    }
}
