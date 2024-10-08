use std::fmt::Debug;
use std::io::Write;
use std::path::PathBuf;
use std::str::FromStr;
use std::{collections::HashMap, path::Path};

use anyhow::{anyhow, bail, Context, Result};
use duckdb::{params, params_from_iter};

use super::{Bindings, TracebackError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::context::context;
use crate::create_pymodule;
use crate::orm::dataset::TextSummarizationDataset;
use crate::orm::{
    ConversationDataset, Hyperparams, Task, TextClassificationDataset,
    TextPairClassificationDataset,
};

pub mod whitelist;

mod transform;
pub use transform::*;

create_pymodule!("/src/bindings/transformers/transformers.py");

// Need a wrapper so we can implement traits for it
pub struct Json(pub Value);

impl From<Json> for Value {
    fn from(value: Json) -> Self {
        value.0
    }
}

impl FromPyObject<'_> for Json {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        if ob.is_instance_of::<PyDict>() {
            let dict: &PyDict = ob.downcast()?;
            let mut json = serde_json::Map::new();
            for (key, value) in dict.iter() {
                let value = Json::extract(value)?;
                json.insert(String::extract(key)?, value.0);
            }
            Ok(Self(serde_json::Value::Object(json)))
        } else if ob.is_instance_of::<PyBool>() {
            let value = bool::extract(ob)?;
            Ok(Self(serde_json::Value::Bool(value)))
        } else if ob.is_instance_of::<PyInt>() {
            let value = i64::extract(ob)?;
            Ok(Self(serde_json::Value::Number(value.into())))
        } else if ob.is_instance_of::<PyFloat>() {
            let value = f64::extract(ob)?;
            let value = serde_json::value::Number::from_f64(value)
                .context("Could not convert f64 to serde_json::Number")?;
            Ok(Self(serde_json::Value::Number(value)))
        } else if ob.is_instance_of::<PyString>() {
            let value = String::extract(ob)?;
            Ok(Self(serde_json::Value::String(value)))
        } else if ob.is_instance_of::<PyList>() {
            let value = ob.downcast::<PyList>()?;
            let mut json_values = Vec::new();
            for v in value {
                let v = v.extract::<Json>()?;
                json_values.push(v.0);
            }
            Ok(Self(serde_json::Value::Array(json_values)))
        } else {
            if ob.is_none() {
                return Ok(Self(serde_json::Value::Null));
            }
            Err(anyhow::anyhow!(
                "Unsupported type for JSON conversion: {:?}",
                ob.get_type()
            ))?
        }
    }
}

pub fn get_model_from(task: &Value) -> Result<String> {
    Python::with_gil(|py| -> Result<String> {
        let get_model_from = get_module!(PY_MODULE)
            .getattr(py, "get_model_from")
            .format_traceback(py)?;
        let model = get_model_from
            .call1(py, PyTuple::new(py, &[task.to_string().into_py(py)]))
            .format_traceback(py)?;
        model.extract(py).format_traceback(py)
    })
}

pub fn embed(
    transformer: &str,
    inputs: Vec<&str>,
    kwargs: &serde_json::Value,
) -> Result<Vec<Vec<f32>>> {
    let kwargs = serde_json::to_string(kwargs)?;
    Python::with_gil(|py| -> Result<Vec<Vec<f32>>> {
        let embed: Py<PyAny> = get_module!(PY_MODULE)
            .getattr(py, "embed")
            .format_traceback(py)?;
        let output = embed
            .call1(
                py,
                PyTuple::new(
                    py,
                    &[
                        transformer.to_string().into_py(py),
                        inputs.into_py(py),
                        kwargs.into_py(py),
                    ],
                ),
            )
            .format_traceback(py)?;

        output.extract(py).format_traceback(py)
    })
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Clone)]
pub struct RankResult {
    pub corpus_id: i64,
    pub score: f64,
    pub text: Option<String>,
}

pub fn rank(
    transformer: &str,
    query: &str,
    documents: Vec<&str>,
    kwargs: &serde_json::Value,
) -> Result<Vec<RankResult>> {
    let kwargs = serde_json::to_string(kwargs)?;
    Python::with_gil(|py| -> Result<Vec<RankResult>> {
        let embed: Py<PyAny> = get_module!(PY_MODULE)
            .getattr(py, "rank")
            .format_traceback(py)?;
        let output = embed
            .call1(
                py,
                PyTuple::new(
                    py,
                    &[
                        transformer.to_string().into_py(py),
                        query.into_py(py),
                        documents.into_py(py),
                        kwargs.into_py(py),
                    ],
                ),
            )
            .format_traceback(py)?;
        let out: Vec<Json> = output.extract(py).format_traceback(py)?;
        out.into_iter()
            .map(|x| {
                let x: RankResult = serde_json::from_value(x.0)?;
                Ok(x)
            })
            .collect()
    })
}

pub struct TextClassifier {
    model_id: i64,
}

impl TextClassifier {
    pub fn from_id(id: i64) -> Result<Box<dyn Bindings>> {
        let result = Python::with_gil(|py| -> Result<Self> {
            let mut dir = std::path::PathBuf::from("/tmp/quackml/models");
            dir.push(id.to_string());
            if !dir.exists() {
                dump_model(id, dir.clone())?;
            }
            let result = context::run(|conn| {
                conn.query_row(
                    "
                        SELECT task::TEXT
                        FROM quackml.projects
                        JOIN quackml.models
                            ON models.project_id = projects.id
                        WHERE models.id = $1
                        ",
                    params![id],
                    |row| row.get::<_, String>(0),
                )
                .map_err(|e| anyhow!("failed to get task: {e}"))
            });
            let task = result.expect("failed to get task");
            let load = get_module!(PY_MODULE).getattr(py, "load_model")?;
            let task =
                Task::from_str(&task).map_err(|_| anyhow!("could not make a Task from {task}"))?;
            load.call1(py, (id, task.to_string(), dir))
                .format_traceback(py)?;
            return Ok(Self { model_id: id });
        });
        result.map(|x| Box::new(x) as Box<dyn Bindings>)
    }
}

impl Debug for TextClassifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextClassifier").finish()
    }
}

impl Bindings for TextClassifier {
    fn predict(
        &self,
        features: &[f32],
        _num_features: usize,
        _num_classes: usize,
    ) -> Result<Vec<f32>> {
        Python::with_gil(|py| -> Result<Vec<f32>> {
            // Convert the features into a string
            let features = features.iter().map(|f| *f as u8).collect::<Vec<_>>();
            let features = vec![String::from_utf8(features)?];
            let features = PyTuple::new(py, &features);
            let module = get_module!(PY_MODULE);
            let predict = module.getattr(py, "predict")?;
            log::info!("Retrieved predict function");
            let output = predict
                .call1(py, (self.model_id, features))
                .format_traceback(py)?;
            log::info!("Called predict function");
            output.extract(py).format_traceback(py)
        })
    }

    fn predict_proba(&self, features: &[f32], num_features: usize) -> Result<Vec<f32>> {
        Python::with_gil(|py| -> Result<Vec<f32>> {
            // Convert the features into a string
            let features = features.iter().map(|f| *f as u8).collect::<Vec<_>>();
            let module = get_module!(PY_MODULE);
            let predict_proba = module.getattr(py, "predict_proba")?;
            let output = predict_proba
                .call1(py, (self.model_id, features, num_features))
                .format_traceback(py)?;
            output.extract(py).format_traceback(py)
        })
    }

    fn to_bytes(&self) -> Result<Vec<u8>> {
        // Python::with_gil(|py| -> Result<Vec<u8>> {
        //     let save = get_module!(PY_MODULE).getattr(py, "save")?;
        //     Ok(save
        //         .call1(py, PyTuple::new(py, [&self.transformer]))?
        //         .extract(py)?)
        // })
        panic!("Not implemented")
    }

    fn from_bytes(bytes: &[u8]) -> Result<Box<dyn Bindings>>
    where
        Self: Sized,
    {
        // let module = get_module!(PY_MODULE);
        // Python::with_gil(|py| -> Result<Box<dyn Bindings>> {
        //     let load = module.getattr(py, "load")?;
        //     let transformer = load
        //         .call1(py, PyTuple::new(py, [bytes]))?
        //         .extract::<Py<PyAny>>(py)?;
        //     let predict = transformer.getattr(py, "predict")?;
        //     let predict_proba = transformer.getattr(py, "predict_proba")?;
        //     Ok(Box::new(TextClassifier {
        //         transformer,
        //         predict,
        //         predict_proba,
        //     }))
        // })
        panic!("Not implemented")
    }
}

pub fn finetune_text_classification(
    task: &Task,
    dataset: TextClassificationDataset,
    hyperparams: &Hyperparams,
    path: &Path,
    project_id: i64,
    model_id: i64,
) -> Result<HashMap<String, f64>> {
    let task = task.to_string();
    let hyperparams = serde_json::to_string(&hyperparams)?;

    Python::with_gil(|py| -> Result<HashMap<String, f64>> {
        let tune = get_module!(PY_MODULE)
            .getattr(py, "finetune_text_classification")
            .format_traceback(py)?;
        let path = path.to_string_lossy();
        let output = tune
            .call1(
                py,
                (
                    &task,
                    &hyperparams,
                    path.as_ref(),
                    dataset.text_train,
                    dataset.text_test,
                    dataset.class_train,
                    dataset.class_test,
                    project_id,
                    model_id,
                ),
            )
            .format_traceback(py)?;

        output.extract(py).format_traceback(py)
    })
}

pub fn finetune_text_pair_classification(
    task: &Task,
    dataset: TextPairClassificationDataset,
    hyperparams: &Hyperparams,
    path: &Path,
    project_id: i64,
    model_id: i64,
) -> Result<HashMap<String, f64>> {
    let task = task.to_string();
    let hyperparams = serde_json::to_string(&hyperparams)?;

    Python::with_gil(|py| -> Result<HashMap<String, f64>> {
        let tune = get_module!(PY_MODULE)
            .getattr(py, "finetune_text_pair_classification")
            .format_traceback(py)?;
        let path = path.to_string_lossy();
        let output = tune
            .call1(
                py,
                (
                    &task,
                    &hyperparams,
                    path.as_ref(),
                    dataset.text1_train,
                    dataset.text1_test,
                    dataset.text2_train,
                    dataset.text2_test,
                    dataset.class_train,
                    dataset.class_test,
                    project_id,
                    model_id,
                ),
            )
            .format_traceback(py)?;

        output.extract(py).format_traceback(py)
    })
}

pub fn finetune_conversation(
    task: &Task,
    dataset: ConversationDataset,
    hyperparams: &Hyperparams,
    path: &Path,
    project_id: i64,
    model_id: i64,
) -> Result<HashMap<String, f64>> {
    let task = task.to_string();
    let hyperparams = serde_json::to_string(&hyperparams)?;

    Python::with_gil(|py| -> Result<HashMap<String, f64>> {
        let tune = get_module!(PY_MODULE)
            .getattr(py, "finetune_conversation")
            .format_traceback(py)?;
        let path = path.to_string_lossy();
        let output = tune
            .call1(
                py,
                (
                    &task,
                    &hyperparams,
                    path.as_ref(),
                    dataset.system_train,
                    dataset.user_test,
                    dataset.assistant_train,
                    dataset.system_test,
                    dataset.user_train,
                    dataset.assistant_test,
                    project_id,
                    model_id,
                ),
            )
            .format_traceback(py)?;

        output.extract(py).format_traceback(py)
    })
}

pub fn finetune_text_summarization(
    task: &Task,
    dataset: TextSummarizationDataset,
    hyperparams: &Hyperparams,
    path: &Path,
    project_id: i64,
    model_id: i64,
) -> Result<HashMap<String, f64>> {
    let task = task.to_string();
    let hyperparams = serde_json::to_string(&hyperparams)?;

    Python::with_gil(|py| -> Result<HashMap<String, f64>> {
        let tune = get_module!(PY_MODULE)
            .getattr(py, "finetune_text_summarization")
            .format_traceback(py)?;
        let path = path.to_string_lossy();
        let output = tune
            .call1(
                py,
                (
                    &task,
                    &hyperparams,
                    path.as_ref(),
                    dataset.text_train,
                    dataset.text_test,
                    dataset.summary_train,
                    dataset.summary_test,
                    project_id,
                    model_id,
                ),
            )
            .format_traceback(py)?;

        output.extract(py).format_traceback(py)
    })
}

pub fn generate(
    model_id: i64,
    inputs: Vec<&str>,
    config: serde_json::Value,
) -> Result<Vec<String>> {
    Python::with_gil(|py| -> Result<Vec<String>> {
        let generate = get_module!(PY_MODULE)
            .getattr(py, "generate")
            .format_traceback(py)?;
        let config = serde_json::to_string(&config)?;
        // cloning inputs in case we have to re-call on error is rather unfortunate here
        // similarly, using a json string to pass kwargs is also unfortunate extra parsing
        // it'd be nice to clean all this up one day
        let result = generate.call1(py, (model_id, inputs.clone(), &config));
        let result = match result {
            Err(e) => {
                if e.get_type(py).name()? == "MissingModelError" {
                    log::info!("Loading model into cache for connection reuse");
                    let mut dir = std::path::PathBuf::from("/tmp/quackml/models");
                    dir.push(model_id.to_string());
                    if !dir.exists() {
                        dump_model(model_id, dir.clone())?;
                    }
                    let result = context::run(|conn| {
                        conn.query_row(
                            "
                        SELECT task::TEXT
                        FROM quackml.projects
                        JOIN quackml.models
                            ON models.project_id = projects.id
                        WHERE models.id = $1
                        ",
                            params![model_id],
                            |row| row.get::<_, String>(0),
                        )
                        .map_err(|e| anyhow!("failed to get task: {e}"))
                    });
                    let task = result.expect("failed to get task");
                    let load = get_module!(PY_MODULE).getattr(py, "load_model")?;
                    let task = Task::from_str(&task)
                        .map_err(|_| anyhow!("could not make a Task from {task}"))?;
                    load.call1(py, (model_id, task.to_string(), dir))
                        .format_traceback(py)?;

                    generate
                        .call1(py, (model_id, inputs, config))
                        .format_traceback(py)?
                } else {
                    return Err(e.into());
                }
            }
            Ok(o) => o,
        };
        result.extract(py).format_traceback(py)
    })
}

fn dump_model(model_id: i64, dir: PathBuf) -> Result<()> {
    if dir.exists() {
        std::fs::remove_dir_all(&dir).context("failed to remove directory while dumping model")?;
    }
    std::fs::create_dir_all(&dir).context("failed to create directory while dumping model")?;
    context::run(|conn| -> Result<()> {
        let mut stmt = conn.prepare("SELECT path, part, data FROM quackml.files WHERE model_id = ? ORDER BY path ASC, part ASC")?;
        let rows = stmt.query_map(params![model_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i32>(1)?,
                row.get::<_, Vec<u8>>(2)?,
            ))
        })?;

        for row_result in rows {
            let (path_str, _, data) = row_result?;
            let mut path = dir.clone();
            path.push(path_str);

            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?;

            file.write_all(&data)?;
            file.flush()?;
        }
        Ok(())
    })
}

pub fn load_dataset(
    name: &str,
    subset: Option<String>,
    limit: Option<usize>,
    kwargs: &serde_json::Value,
) -> Result<usize> {
    let kwargs = serde_json::to_string(kwargs)?;

    let dataset = Python::with_gil(|py| -> Result<String> {
        let load_dataset: Py<PyAny> = get_module!(PY_MODULE)
            .getattr(py, "load_dataset")
            .format_traceback(py)?;
        load_dataset
            .call1(
                py,
                PyTuple::new(
                    py,
                    &[
                        name.into_py(py),
                        subset.into_py(py),
                        limit.into_py(py),
                        kwargs.into_py(py),
                    ],
                ),
            )
            .format_traceback(py)?
            .extract(py)
            .format_traceback(py)
    })?;

    let table_name = format!("quackml.\"{}\"", name);

    // Columns are a (name: String, values: Vec<Value>) pair
    let json: serde_json::Value = serde_json::from_str(&dataset)?;
    let json = json
        .as_object()
        .ok_or(anyhow!("dataset json is not object"))?;
    let types = json
        .get("types")
        .ok_or(anyhow!("dataset json missing `types` key"))?
        .as_object()
        .ok_or(anyhow!("dataset `types` key is not an object"))?;
    let data = json
        .get("data")
        .ok_or(anyhow!("dataset json missing `data` key"))?
        .as_object()
        .ok_or(anyhow!("dataset `data` key is not an object"))?;
    let column_names = types
        .iter()
        .map(|(name, _type)| format!("\"{}\"", name))
        .collect::<Vec<String>>()
        .join(", ");
    let column_types = types
        .iter()
        .map(|(name, type_)| -> Result<String> {
            let type_ = type_
                .as_str()
                .ok_or(anyhow!("expected {type_} to be a json string"))?;
            let type_ = match type_ {
                "string" => "TEXT",
                "dict" | "list" => "JSONB",
                "int64" => "INT8",
                "int32" => "INT4",
                "int16" => "INT2",
                "int8" => "INT2",
                "float64" => "FLOAT8",
                "float32" => "FLOAT4",
                "float16" => "FLOAT4",
                "bool" => "BOOLEAN",
                _ => bail!("unhandled dataset feature while reading dataset: {type_}"),
            };
            Ok(format!("\"{name}\" {type_}"))
        })
        .collect::<Result<Vec<String>>>()?
        .join(", ");
    let column_placeholders = types
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let placeholder = i + 1;
            format!("${placeholder}")
        })
        .collect::<Vec<String>>()
        .join(", ");
    let num_cols = types.len();
    let num_rows = data
        .values()
        .next()
        .ok_or(anyhow!("dataset json has no fields"))?
        .as_array()
        .ok_or(anyhow!("dataset json field is not an array"))?
        .len();
    // Avoid the existence warning by checking the schema for the table first
    let table_count: i64 = context::run(|conn| {
        conn.query_row(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ? AND table_schema = 'quackml'",
            params![table_name.clone()],
            |row| row.get(0),
        ).map_err(|e| anyhow!("failed to check for table existence: {e}"))
    })?;

    if table_count == 1 {
        context::run(|conn| {
            conn.execute(&format!("DROP TABLE IF EXISTS {}", table_name), [])
                .map_err(|e| anyhow!("failed to drop table: {e}"))
        })?;
    }

    context::run(|conn| {
        conn.execute(
            &format!("CREATE TABLE {} ({})", table_name, column_types),
            [],
        )
        .map_err(|e| anyhow!("failed to create table: {e}"))
    })?;

    let insert = format!(
        "INSERT INTO {} ({}) VALUES ({})",
        table_name, column_names, column_placeholders
    );

    for i in 0..num_rows {
        let mut row = Vec::with_capacity(num_cols);
        for (name, values) in data {
            let value = values
                .as_array()
                .ok_or_else(|| anyhow!("expected {values} to be an array"))?
                .get(i)
                .ok_or_else(|| anyhow!("invalid index {i} for {values}"))?;
            match types
                .get(name)
                .ok_or_else(|| anyhow!("{types:?} expected to have key {name}"))?
                .as_str()
                .ok_or_else(|| anyhow!("json field {name} expected to be string"))?
            {
                "string" => row.push(
                    value
                        .as_str()
                        .ok_or_else(|| anyhow!("expected {value} to be string"))?
                        .to_string(),
                ),
                "dict" | "list" => row.push(serde_json::to_string(value)?),
                "int64" | "int32" | "int16" | "int8" => row.push(
                    value
                        .as_i64()
                        .ok_or_else(|| anyhow!("expected {value} to be i64"))?
                        .to_string(),
                ),
                "float64" | "float32" | "float16" => row.push(
                    value
                        .as_f64()
                        .ok_or_else(|| anyhow!("expected {value} to be f64"))?
                        .to_string(),
                ),
                "bool" => row.push(
                    value
                        .as_bool()
                        .ok_or_else(|| anyhow!("expected {value} to be bool"))?
                        .to_string(),
                ),
                type_ => {
                    bail!("unhandled dataset value type while reading dataset: {value:?} {type_:?}")
                }
            }
        }
        context::run(|conn| {
            conn.execute(&insert, params_from_iter(row.iter()))
                .map_err(|e| anyhow!("failed to insert row: {e}"))
        })?;
    }

    Ok(num_rows)
}

pub fn clear_gpu_cache(memory_usage: Option<f32>) -> Result<bool> {
    Python::with_gil(|py| -> Result<bool> {
        let clear_gpu_cache: Py<PyAny> = get_module!(PY_MODULE)
            .getattr(py, "clear_gpu_cache")
            .format_traceback(py)?;
        let success = clear_gpu_cache
            .call1(py, PyTuple::new(py, &[memory_usage.into_py(py)]))
            .format_traceback(py)?
            .extract(py)
            .format_traceback(py)?;
        Ok(success)
    })
}
