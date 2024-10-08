use core::{f32, f64};
use std::ffi::{c_char, c_float};
use std::fmt::Write;
use std::io::ErrorKind;
use std::result;
use std::str::FromStr;
use std::{default, ffi::CString};

use anyhow::{anyhow, Error};
use duckdb::core::{DataChunkHandle, ListVector};
use duckdb::polars::chunked_array::collect;
use duckdb::vtab::{FunctionInfo, InitInfo, VScalar};
use duckdb::{
    core::{Inserter, LogicalTypeHandle, LogicalTypeId},
    params,
    vtab::{Free, VTab},
    Rows,
};
use itertools::izip;
use libduckdb_sys::{
    duckdb_create_list_type, duckdb_list_entry, duckdb_string_t, duckdb_vector, DuckDBSuccess,
};
use log::*;
use ndarray::{AssignElem, Zip};

#[cfg(feature = "python")]
use serde_json::json;
use serde_json::{Map, Value};

use crate::context::context;
#[cfg(feature = "python")]
use crate::orm::*;

macro_rules! unwrap_or_error {
    ($i:expr) => {
        match $i {
            Ok(v) => v,
            Err(e) => panic!("{e}"),
        }
    };
}

#[cfg(feature = "python")]
pub fn activate_venv(venv: &str) -> bool {
    let venv = match venv {
        "" => None,
        _ => Some(venv),
    };
    unwrap_or_error!(crate::bindings::python::activate_venv(&venv))
}

#[cfg(feature = "python")]
pub fn validate_python_dependencies() -> bool {
    unwrap_or_error!(crate::bindings::python::validate_dependencies())
}

#[cfg(not(feature = "python"))]
pub fn validate_python_dependencies() {}

#[cfg(feature = "python")]
pub fn python_package_version(name: &str) -> String {
    unwrap_or_error!(crate::bindings::python::package_version(name))
}

#[cfg(feature = "python")]
pub fn python_version() -> String {
    unwrap_or_error!(crate::bindings::python::version())
}

#[cfg(not(feature = "python"))]
pub fn python_package_version(name: &str) {
    error!("Python is not installed, recompile with `--features python`");
}

#[cfg(feature = "python")]
pub fn python_pip_freeze() -> Vec<String> {
    unwrap_or_error!(crate::bindings::python::pip_freeze())
}

#[cfg(not(feature = "python"))]
pub fn python_version() -> String {
    String::from("Python is not installed, recompile with `--features python`")
}

// pub fn validate_shared_library() {
//     let shared_preload_libraries = context::run(|conn| {
//         conn.query_row(
//             "SELECT setting
//          FROM pg_settings
//          WHERE name = 'shared_preload_libraries'
//          LIMIT 1",
//             params![],
//             |row| row.get(0)?,
//         )
//         .map_err(|| anyhow!("Error getting setting"))
//     });
//     let shared_preload_libraries: String = Spi::get_one(
//         "SELECT setting
//          FROM pg_settings
//          WHERE name = 'shared_preload_libraries'
//          LIMIT 1",
//     )
//     .unwrap()
//     .unwrap();
//
//     if !shared_preload_libraries.contains("quackml") {
//         error!("`quackml` must be added to `shared_preload_libraries` setting or models cannot be deployed");
//     }
// }
//
fn version() -> String {
    // format!("{} ({})", crate::VERSION, crate::COMMIT)
    "0.0.0".to_string()
}

#[repr(C)]
pub struct TrainBindData {
    project_name: *mut c_char,
    task: *mut c_char,
    relation_name: *mut c_char,
    y_column_name: *mut c_char,
    algorithm: *mut c_char,
    hyperparams: *mut c_char,
    search: *mut c_char,
    search_params: *mut c_char,
    search_args: *mut c_char,
    test_size: *mut c_float,
    test_sampling: *mut c_char,
    preprocess: *mut c_char,
}

impl Free for TrainBindData {
    fn free(&mut self) {
        unsafe {
            if !self.project_name.is_null() {
                drop(CString::from_raw(self.project_name));
            }
            if !self.task.is_null() {
                drop(CString::from_raw(self.task));
            }
            if !self.relation_name.is_null() {
                drop(CString::from_raw(self.relation_name));
            }
            if !self.y_column_name.is_null() {
                drop(CString::from_raw(self.y_column_name));
            }
            if !self.algorithm.is_null() {
                drop(CString::from_raw(self.algorithm));
            }
            if !self.hyperparams.is_null() {
                drop(CString::from_raw(self.hyperparams));
            }
            if !self.search.is_null() {
                drop(CString::from_raw(self.search));
            }
            if !self.search_params.is_null() {
                drop(CString::from_raw(self.search_params));
            }
            if !self.search_args.is_null() {
                drop(CString::from_raw(self.search_args));
            }
            if !self.test_sampling.is_null() {
                drop(CString::from_raw(self.test_sampling));
            }
            if !self.preprocess.is_null() {
                drop(CString::from_raw(self.preprocess));
            }
        }
    }
}

#[repr(C)]
pub struct TrainInitData {
    done: bool,
}

impl Free for TrainInitData {
    fn free(&mut self) {}
}
pub struct TrainVTab;

impl VTab for TrainVTab {
    type InitData = TrainInitData;
    type BindData = TrainBindData;

    unsafe fn bind(
        bind: &duckdb::vtab::BindInfo,
        data: *mut Self::BindData,
    ) -> duckdb::Result<(), Box<dyn std::error::Error>> {
        bind.add_result_column("project", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("task", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("algorithm", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("deploy", LogicalTypeHandle::from(LogicalTypeId::Boolean));

        let project_name = bind.get_parameter(0).to_string();
        let task = bind
            .get_named_parameter("task")
            .map_or_else(|| "".to_string(), |v| v.to_string());
        let relation_name = bind
            .get_named_parameter("relation_name")
            .map_or_else(|| "".to_string(), |v| v.to_string());
        let y_column_name = bind
            .get_named_parameter("y_column_name")
            .map_or_else(|| "".to_string(), |v| v.to_string());
        let algorithm = bind
            .get_named_parameter("algorithm")
            .map_or_else(|| "".to_string(), |v| v.to_string());
        let hyperparams = bind
            .get_named_parameter("hyperparams")
            .map_or_else(|| "".to_string(), |v| v.to_string());
        let search = bind
            .get_named_parameter("search")
            .map_or_else(|| "".to_string(), |v| v.to_string());
        let search_params = bind
            .get_named_parameter("search_params")
            .map_or_else(|| "".to_string(), |v| v.to_string());
        let search_args = bind
            .get_named_parameter("search_args")
            .map_or_else(|| "".to_string(), |v| v.to_string());
        let test_size = bind
            .get_named_parameter("test_size")
            .map_or_else(|| 0.0, |v| v.to_string().parse::<f32>().unwrap_or(0.0));
        let test_sampling = bind
            .get_named_parameter("test_sampling")
            .map_or_else(|| "".to_string(), |v| v.to_string());
        let preprocess = bind
            .get_named_parameter("preprocess")
            .map_or_else(|| "".to_string(), |v| v.to_string());

        unsafe {
            (*data).project_name = CString::new(project_name).unwrap().into_raw();
            (*data).task = CString::new(task).unwrap().into_raw();
            (*data).relation_name = CString::new(relation_name).unwrap().into_raw();
            (*data).y_column_name = CString::new(y_column_name).unwrap().into_raw();
            (*data).algorithm = CString::new(algorithm).unwrap().into_raw();
            (*data).hyperparams = CString::new(hyperparams).unwrap().into_raw();
            (*data).search = CString::new(search).unwrap().into_raw();
            (*data).search_params = CString::new(search_params).unwrap().into_raw();
            (*data).search_args = CString::new(search_args).unwrap().into_raw();
            (*data).test_size = Box::into_raw(Box::new(test_size));
            (*data).test_sampling = CString::new(test_sampling).unwrap().into_raw();
            (*data).preprocess = CString::new(preprocess).unwrap().into_raw();
        }
        Ok(())
    }

    unsafe fn init(
        init: &duckdb::vtab::InitInfo,
        data: *mut Self::InitData,
    ) -> duckdb::Result<(), Box<dyn std::error::Error>> {
        unsafe {
            (*data).done = false;
        }
        Ok(())
    }

    unsafe fn func(
        func: &duckdb::vtab::FunctionInfo,
        output: &mut duckdb::core::DataChunkHandle,
    ) -> duckdb::Result<(), Box<dyn std::error::Error>> {
        let init_info = func.get_init_data::<TrainInitData>();
        let bind_info = func.get_bind_data::<TrainBindData>();

        unsafe {
            if (*init_info).done {
                output.set_len(0)
            } else {
                (*init_info).done = true;
                let project_name = CString::from_raw((*bind_info).project_name);
                let task = CString::from_raw((*bind_info).task);
                let relation_name = CString::from_raw((*bind_info).relation_name);
                let y_column_name = CString::from_raw((*bind_info).y_column_name);
                let algorithm = CString::from_raw((*bind_info).algorithm);
                let hyperparams = CString::from_raw((*bind_info).hyperparams);
                let search = CString::from_raw((*bind_info).search);
                let search_params = CString::from_raw((*bind_info).search_params);
                let search_args = CString::from_raw((*bind_info).search_args);
                let test_sampling = CString::from_raw((*bind_info).test_sampling);
                let preprocess = CString::from_raw((*bind_info).preprocess);

                (*bind_info).project_name = CString::into_raw(project_name.clone());
                (*bind_info).task = CString::into_raw(task.clone());
                (*bind_info).relation_name = CString::into_raw(relation_name.clone());
                (*bind_info).y_column_name = CString::into_raw(y_column_name.clone());
                (*bind_info).algorithm = CString::into_raw(algorithm.clone());
                (*bind_info).hyperparams = CString::into_raw(hyperparams.clone());
                (*bind_info).search = CString::into_raw(search.clone());
                (*bind_info).search_params = CString::into_raw(search_params.clone());
                (*bind_info).search_args = CString::into_raw(search_args.clone());
                (*bind_info).test_sampling = CString::into_raw(test_sampling.clone());
                (*bind_info).preprocess = CString::into_raw(preprocess.clone());

                let task = match task.to_str() {
                    Ok("") => None,
                    Ok(s) => Some(s),
                    Err(_) => panic!("Failed to unwrap task string"),
                };
                let relation_name = match relation_name.to_str() {
                    Ok("") => None,
                    Ok(s) => Some(s),
                    Err(_) => panic!("Failed to unwrap relation_name string"),
                };
                let y_column_name = match y_column_name.to_str() {
                    Ok("") => None,
                    Ok(s) => Some(s),
                    Err(_) => panic!("Failed to unwrap y_column_name string"),
                };
                let algorithm = match algorithm.to_str() {
                    Ok("") => None,
                    Ok(s) => Some(Algorithm::from_str(s).unwrap()),
                    Err(_) => panic!("Failed to unwrap algorithm string"),
                };
                let hyperparams = match hyperparams.to_str() {
                    Ok("") => None,
                    Ok(s) => Some(serde_json::from_str(s).unwrap()),
                    Err(_) => panic!("Failed to unwrap hyperparams string"),
                };
                let search = match search.to_str() {
                    Ok("") => None,
                    Ok(s) => Some(Search::from_str(s).unwrap()),
                    Err(_) => panic!("Failed to unwrap search string"),
                };
                let search_params = match search_params.to_str() {
                    Ok("") => None,
                    Ok(s) => Some(serde_json::from_str(s).unwrap()),
                    Err(_) => panic!("Failed to unwrap search_params string"),
                };
                let search_args = match search_args.to_str() {
                    Ok("") => None,
                    Ok(s) => Some(serde_json::from_str(s).unwrap()),
                    Err(_) => panic!("Failed to unwrap search_args string"),
                };
                let test_size = match unsafe { *(*bind_info).test_size } {
                    0.0 => None,
                    value => Some(value),
                };
                let test_sampling = match test_sampling.to_str() {
                    Ok("") => None,
                    Ok(s) => Some(Sampling::from_str(s).unwrap()),
                    Err(_) => panic!("Failed to unwrap test_sampling string"),
                };
                let preprocess = match preprocess.to_str() {
                    Ok("") => None,
                    Ok(s) => Some(serde_json::from_str(s).unwrap()),
                    Err(_) => panic!("Failed to unwrap preprocess string"),
                };
                let result = train(
                    project_name.to_str().unwrap(),
                    task,
                    relation_name,
                    y_column_name,
                    algorithm,
                    hyperparams,
                    search,
                    search_params,
                    search_args,
                    test_size,
                    test_sampling,
                    None,
                    None,
                    preprocess,
                );

                let proj_column = output.flat_vector(0);
                let task_column = output.flat_vector(1);
                let algorithm_column = output.flat_vector(2);
                let deploy_column: *mut bool = output.flat_vector(3).as_mut_ptr();
                let project_name_raw = CString::new(result.project_name).unwrap();
                let task_raw = CString::new(result.task).unwrap();
                let algorithm_raw = CString::new(result.algorithm).unwrap();

                proj_column.insert(0, project_name_raw);
                task_column.insert(0, task_raw);
                algorithm_column.insert(0, algorithm_raw);
                deploy_column.write(result.deploy);
                output.set_len(1);
            }
        }
        Ok(())
    }

    fn parameters() -> Option<Vec<duckdb::core::LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }

    fn named_parameters() -> Option<Vec<(String, duckdb::core::LogicalTypeHandle)>> {
        Some(vec![
            (
                "task".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
            (
                "relation_name".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
            (
                "y_column_name".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
            (
                "algorithm".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
            (
                "hyperparams".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
            (
                "search".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
            (
                "search_params".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
            (
                "search_args".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
            (
                "test_size".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Float),
            ),
            (
                "test_sampling".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
            (
                "preprocess".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
        ])
    }
}

struct TrainResult {
    project_name: String,
    task: String,
    algorithm: String,
    deploy: bool,
}

#[allow(clippy::too_many_arguments)]
fn train(
    project_name: &str,
    task: Option<&str>,
    relation_name: Option<&str>,
    y_column_name: Option<&str>,
    algorithm: Option<Algorithm>,
    hyperparams: Option<Hyperparams>,
    search: Option<Search>,
    search_params: Option<Value>,
    search_args: Option<Value>,
    test_size: Option<f32>,
    test_sampling: Option<Sampling>,
    automatic_deploy: Option<bool>,
    materialize_snapshot: Option<bool>,
    preprocess: Option<Value>,
) -> TrainResult {
    let task = task.unwrap_or("NULL");
    let relation_name = relation_name.unwrap_or("NULL");
    let y_column_name = y_column_name.map(|y_column_name| vec![y_column_name.to_string()]);
    let algorithm = algorithm.unwrap_or(Algorithm::linear);
    let hyperparams = hyperparams.unwrap_or_else(|| serde_json::Map::new());
    let search_params = search_params
        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()))
        .to_string();
    let search_args = search_args
        .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()))
        .to_string();
    let test_size = test_size.unwrap_or(0.25);
    let test_sampling = test_sampling.unwrap_or(Sampling::stratified);
    let materialize_snapshot = materialize_snapshot.unwrap_or(false);
    let preprocess = preprocess.unwrap_or("{}".into()).to_string();

    train_joint(
        project_name,
        Some(task),
        Some(relation_name),
        y_column_name,
        algorithm,
        &hyperparams,
        search,
        search_params,
        search_args,
        test_size,
        test_sampling,
        automatic_deploy,
        materialize_snapshot,
        preprocess,
    )
}

#[allow(clippy::too_many_arguments)]
fn train_joint(
    project_name: &str,
    task: Option<&str>,
    relation_name: Option<&str>,
    y_column_name: Option<Vec<String>>,
    algorithm: Algorithm,
    hyperparams: &Map<String, Value>,
    search: Option<Search>,
    search_params: String,
    search_args: String,
    test_size: f32,
    test_sampling: Sampling,
    automatic_deploy: Option<bool>,
    materialize_snapshot: bool,
    preprocess: String,
) -> TrainResult {
    let task = task.map(|t| Task::from_str(t).unwrap());
    let project = match Project::find_by_name(project_name) {
        Some(project) => project,
        None => Project::create(
            project_name,
            match task {
                Some(task) => task,
                None => panic!(
                    "Project `{}` does not exist. To create a new project, you must specify a `task`.",
                    project_name
                ),
            },
        ),
    };

    if task.is_some() && task.unwrap() != project.task {
        error!(
            "Project `{:?}` already exists with a different task: `{:?}`. Create a new project instead.",
            project.name, project.task
        );
    }

    let mut snapshot = match relation_name {
        None => {
            let snapshot = project.last_snapshot().expect(
                "You must pass a `relation_name` and `y_column_name` to snapshot the first time you train a model.",
            );

            info!("Using existing snapshot from {}", snapshot.snapshot_name(),);

            snapshot
        }

        Some(relation_name) => {
            info!(
                "Snapshotting table \"{}\", this may take a little while...",
                relation_name
            );

            if project.task.is_supervised() && y_column_name.is_none() {
                error!("You must pass a `y_column_name` when you pass a `relation_name` for a supervised task.");
            }

            let snapshot = Snapshot::create(
                relation_name,
                y_column_name,
                test_size,
                test_sampling,
                materialize_snapshot,
                &preprocess,
            );

            if materialize_snapshot {
                info!(
                    "Snapshot of table \"{}\" created and saved in {}",
                    relation_name,
                    snapshot.snapshot_name(),
                );
            }

            snapshot
        }
    };

    // fix up default algorithm for clustering
    let algorithm = if algorithm == Algorithm::linear && project.task == Task::clustering {
        Algorithm::kmeans
    } else if algorithm == Algorithm::linear && project.task == Task::decomposition {
        Algorithm::pca
    } else {
        algorithm
    };

    println!("Creating model...");
    // # Default repeatable random state when possible
    // let algorithm = Model.algorithm_from_name_and_task(algorithm, task);
    // if "random_state" in algorithm().get_params() and "random_state" not in hyperparams:
    //     hyperparams["random_state"] = 0
    let model = Model::create(
        &project,
        &mut snapshot,
        algorithm,
        hyperparams,
        search,
        serde_json::from_str(&search_params).unwrap(),
        serde_json::from_str(&search_args).unwrap(),
    )
    .unwrap();

    let new_metrics: &serde_json::Value = &model.metrics.expect("Failed to get model metrics");

    let new_metrics = new_metrics.as_object().unwrap();

    let deployed_metrics = context::run(|conn| {
        conn.query_row(
            "
            SELECT models.metrics
            FROM quackml.models
            JOIN quackml.deployments
                ON deployments.model_id = models.id
            JOIN quackml.projects
                ON projects.id = deployments.project_id
            WHERE projects.name = $1
            ORDER by deployments.created_at DESC
            LIMIT 1;
            ",
            params![project_name],
            |row| row.get::<_, String>(0),
        )
        .map_err(|e| anyhow!("Failed to get deployed metrics: {}", e))
    });

    let mut deploy = true;

    println!("Automatic deploy: {:?}", automatic_deploy);
    match automatic_deploy {
        // Deploy only if metrics are better than previous model, or if its the first model
        Some(true) | None => {
            if let Ok(deployed_metrics) = deployed_metrics {
                if let Some(deployed_metrics_obj) =
                    serde_json::from_str::<serde_json::Value>(&deployed_metrics).ok()
                {
                    let default_target_metric = project.task.default_target_metric();
                    let deployed_metric = deployed_metrics_obj
                        .get(&default_target_metric)
                        .and_then(|v| v.as_f64());
                    info!(
                        "Comparing to deployed model {}: {:?}",
                        default_target_metric, deployed_metric
                    );
                    let new_metric = new_metrics
                        .get(&default_target_metric)
                        .and_then(|v| v.as_f64());

                    match (deployed_metric, new_metric) {
                        (Some(deployed), Some(new)) => {
                            // only compare metrics when both new and old model have metrics to compare
                            if project.task.value_is_better(deployed, new) {
                                log::info!(
                                    "New model's {} is not better than current model. New: {}, Current {}",
                                    &default_target_metric,
                                    new,
                                    deployed
                                );
                                deploy = false;
                            }
                        }
                        (None, None) => {
                            info!("No metrics available for both deployed and new model. Deploying new model.")
                        }
                        (Some(_deployed), None) => {
                            info!("No metrics for new model. Retaining old model.");
                            deploy = false;
                        }
                        (None, Some(_new)) => {
                            info!("No metrics for deployed model. Deploying new model.")
                        }
                    }
                } else {
                    info!("Failed to parse deployed model metrics. Check data types of model metadata on quackml.models.metrics");
                    deploy = false;
                }
            }
        }
        Some(false) => {
            info!("Automatic deployment disabled via configuration.");
            deploy = false;
        }
    };

    if deploy {
        project.deploy(model.id, Strategy::new_score);
    } else {
        info!("Not deploying newly trained model.");
    }

    TrainResult {
        project_name: project.name,
        task: project.task.to_string(),
        algorithm: model.algorithm.to_string(),
        deploy,
    }
}

fn deploy_model(model_id: i64) -> Vec<(String, String, String)> {
    let model = unwrap_or_error!(Model::find_cached(model_id));

    let project_id = context::run(|conn| {
        conn.query_row(
            "SELECT projects.id from quackml.projects JOIN pgml.models ON models.project_id = projects.id WHERE models.id = $1",
            params![model_id],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|e| anyhow!("Failed to get project id: {}", e))
    }).expect("Failed to get project id");

    let project = Project::find(project_id).unwrap();
    project.deploy(model_id, Strategy::specific);

    vec![(
        project.name,
        Strategy::specific.to_string(),
        model.algorithm.to_string(),
    )]
}

fn deploy_strategy(
    project_name: &str,
    strategy: Strategy,
    algorithm: Option<Algorithm>,
) -> Vec<(String, String, String)> {
    let (project_id, task) = context::run(|conn| {
        conn.query_row(
            "SELECT id, task::TEXT from quackml.projects WHERE name = $1",
            params![project_name],
            |row| {
                Ok((
                    row.get::<_, i64>(0).unwrap(),
                    row.get::<_, String>(1).unwrap(),
                ))
            },
        )
        .map_err(|e| anyhow!("Failed to get project id: {}", e))
    })
    .expect("Failed to get project id");

    let task = Task::from_str(&task).unwrap();

    let mut sql = "SELECT models.id, models.algorithm::TEXT FROM quackml.models JOIN pgml.projects ON projects.id = models.project_id".to_string();
    let mut predicate = "\nWHERE projects.name = $1".to_string();
    if let Some(algorithm) = algorithm {
        let _ = write!(
            predicate,
            "\nAND algorithm::TEXT = '{}'",
            algorithm.to_string().as_str()
        );
    }
    match strategy {
        Strategy::best_score => {
            let _ = write!(
                sql,
                "{predicate}\n{}",
                task.default_target_metric_sql_order()
            );
        }

        Strategy::most_recent => {
            let _ = write!(sql, "{predicate}\nORDER by models.created_at DESC");
        }

        Strategy::rollback => {
            let _ = write!(
                sql,
                "
                JOIN quackml.deployments ON deployments.project_id = projects.id
                    AND deployments.model_id = models.id
                    AND models.id != (
                        SELECT deployments.model_id
                        FROM quackml.deployments
                        JOIN quackml.projects
                            ON projects.id = deployments.project_id
                        WHERE projects.name = $1
                        ORDER by deployments.created_at DESC
                        LIMIT 1
                    )
                {predicate}
                ORDER by deployments.created_at DESC
            "
            );
        }
        _ => error!("invalid strategy"),
    }
    sql += "\nLIMIT 1";
    let (model_id, algorithm) = context::run(|conn| {
        conn.query_row(&sql, params![project_name], |row| {
            Ok((
                row.get::<_, i64>(0).unwrap(),
                row.get::<_, String>(1).unwrap(),
            ))
        })
        .map_err(|e| anyhow!("Failed to get model id: {}", e))
    })
    .expect("Failed to get model id");

    let project = Project::find(project_id).unwrap();
    project.deploy(model_id, strategy);

    vec![(project_name.to_string(), strategy.to_string(), algorithm)]
}

pub struct PredictScalar {}

impl VScalar for PredictScalar {
    unsafe fn func(
        func: &duckdb::vtab::FunctionInfo,
        input: &mut duckdb::core::DataChunkHandle,
        output: &mut duckdb::core::FlatVector,
    ) -> duckdb::Result<(), Box<dyn std::error::Error>> {
        let rows = input.len();
        let binding = input.flat_vector(0);
        let binding: &[duckdb_string_t] = binding.as_slice::<duckdb_string_t>();
        let project_names = binding
            .iter()
            .take(rows)
            .map(|s| String::from(s))
            .collect::<Vec<String>>();
        let binding = input.list_vector(1);
        let features = binding.to_vec::<f32>(rows);
        if features.iter().map(|v| v.len()).any(|l| l == 0) {
            let error = std::io::Error::new(ErrorKind::InvalidInput, "Features cannot be empty");
            return Err(Box::new(error));
        }
        if features.iter().any(|v| v.iter().any(|o| o.is_none())) {
            let error = std::io::Error::new(
                ErrorKind::InvalidInput,
                "Feature vector contains None values",
            );
            return Err(Box::new(error));
        }
        let features = features
            .iter()
            .map(|v| v.iter().map(|o| o.unwrap()).collect::<Vec<f32>>())
            .collect::<Vec<Vec<f32>>>();

        let projects_features = izip!(project_names, features);
        let results = projects_features
            .map(|(project_name, feature)| predict_f32(project_name.as_str(), feature))
            .collect::<Vec<f32>>();
        let output = output.as_mut_slice::<f32>();
        output[..results.len()].copy_from_slice(&results);
        Ok(())
    }

    fn parameters() -> Option<Vec<duckdb::core::LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Float)),
        ])
    }

    fn return_type() -> LogicalTypeHandle {
        LogicalTypeHandle::from(LogicalTypeId::Float)
    }
}

pub struct PredictProbaScalar {}

impl VScalar for PredictProbaScalar {
    unsafe fn func(
        func: &duckdb::vtab::FunctionInfo,
        input: &mut duckdb::core::DataChunkHandle,
        output: &mut duckdb::core::FlatVector,
    ) -> duckdb::Result<(), Box<dyn std::error::Error>> {
        let rows = input.len();
        let binding = input.flat_vector(0);
        let binding: &[duckdb_string_t] = binding.as_slice::<duckdb_string_t>();
        let project_names = binding
            .iter()
            .take(rows)
            .map(|s| String::from(s))
            .collect::<Vec<String>>();
        let binding = input.list_vector(1);
        let features = binding.to_vec::<f32>(rows);
        if features.iter().map(|v| v.len()).any(|l| l == 0) {
            let error = std::io::Error::new(ErrorKind::InvalidInput, "Features cannot be empty");
            return Err(Box::new(error));
        }
        if features.iter().any(|v| v.iter().any(|o| o.is_none())) {
            let error = std::io::Error::new(
                ErrorKind::InvalidInput,
                "Feature vector contains None values",
            );
            return Err(Box::new(error));
        }
        let features = features
            .iter()
            .map(|v| v.iter().map(|o| o.unwrap()).collect::<Vec<f32>>())
            .collect::<Vec<Vec<f32>>>();
        let projects_features = izip!(project_names, features);
        let results = projects_features
            .map(|(project_name, feature)| predict_proba(project_name.as_str(), feature))
            .collect::<Vec<Vec<f32>>>();
        let output = output.as_mut_slice::<f32>();
        let mut i = 0;
        for result in results {
            output[i] = result[0];
            i += 1;
        }
        Ok(())
    }
    fn parameters() -> Option<Vec<duckdb::core::LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Float)),
        ])
    }
    fn return_type() -> LogicalTypeHandle {
        LogicalTypeHandle::from(LogicalTypeId::Float)
    }
}

pub struct PredictStringScalar {}

impl VScalar for PredictStringScalar {
    unsafe fn func(
        func: &duckdb::vtab::FunctionInfo,
        input: &mut duckdb::core::DataChunkHandle,
        output: &mut duckdb::core::FlatVector,
    ) -> duckdb::Result<(), Box<dyn std::error::Error>> {
        let rows = input.len();
        let binding = input.flat_vector(0);
        let binding: &[duckdb_string_t] = binding.as_slice::<duckdb_string_t>();
        let project_names = binding
            .iter()
            .take(rows)
            .map(|s| String::from(s))
            .collect::<Vec<String>>();
        let binding = input.flat_vector(1);
        let feature_strings: Vec<String> = binding
            .as_slice::<duckdb_string_t>()
            .iter()
            .take(rows)
            .map(|s| String::from(s))
            .collect();

        let features = feature_strings
            .iter()
            .map(|s| s.as_bytes())
            .map(|b| b.iter().map(|byte| *byte as f32).collect::<Vec<f32>>())
            .collect::<Vec<Vec<f32>>>();

        if features.iter().any(|v| v.is_empty()) {
            return Err(Box::new(std::io::Error::new(
                ErrorKind::InvalidInput,
                "Features cannot be empty",
            )));
        }

        let projects_features = izip!(project_names, features);
        let results = projects_features
            .map(|(project_name, feature)| predict_f32(project_name.as_str(), feature))
            .collect::<Vec<f32>>();
        let output = output.as_mut_slice::<f32>();
        let mut i = 0;
        for result in results {
            output[i] = result;
            i += 1;
        }
        Ok(())
    }
    fn parameters() -> Option<Vec<duckdb::core::LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        ])
    }
    fn return_type() -> LogicalTypeHandle {
        LogicalTypeHandle::from(LogicalTypeId::Float)
    }
}

fn predict_f32(project_name: &str, features: Vec<f32>) -> f32 {
    predict_model(Project::get_deployed_model_id(project_name), features)
}

fn predict_f64(project_name: &str, features: Vec<f64>) -> f32 {
    predict_f32(project_name, features.iter().map(|&i| i as f32).collect())
}

fn predict_i16(project_name: &str, features: Vec<i16>) -> f32 {
    predict_f32(project_name, features.iter().map(|&i| i as f32).collect())
}

fn predict_i32(project_name: &str, features: Vec<i32>) -> f32 {
    predict_f32(project_name, features.iter().map(|&i| i as f32).collect())
}

fn predict_i64(project_name: &str, features: Vec<i64>) -> f32 {
    predict_f32(project_name, features.iter().map(|&i| i as f32).collect())
}

fn predict_bool(project_name: &str, features: Vec<bool>) -> f32 {
    predict_f32(
        project_name,
        features.iter().map(|&i| i as u8 as f32).collect(),
    )
}

fn predict_proba(project_name: &str, features: Vec<f32>) -> Vec<f32> {
    predict_model_proba(Project::get_deployed_model_id(project_name), features)
}

fn predict_joint(project_name: &str, features: Vec<f32>) -> Vec<f32> {
    predict_model_joint(Project::get_deployed_model_id(project_name), features)
}

fn predict_batch(project_name: &str, features: Vec<f32>) -> Vec<f32> {
    predict_model_batch(Project::get_deployed_model_id(project_name), features)
}

fn decompose(project_name: &str, vector: Vec<f32>) -> Vec<f32> {
    let model_id = Project::get_deployed_model_id(project_name);
    let model = unwrap_or_error!(Model::find_cached(model_id));
    unwrap_or_error!(model.decompose(&vector))
}

fn predict_row(project_name: &str, row: &mut Rows) -> f32 {
    predict_model_row(Project::get_deployed_model_id(project_name), row)
}

fn predict_model(model_id: i64, features: Vec<f32>) -> f32 {
    let model = unwrap_or_error!(Model::find_cached(model_id));
    unwrap_or_error!(model.predict(&features))
}

fn predict_model_proba(model_id: i64, features: Vec<f32>) -> Vec<f32> {
    let model = unwrap_or_error!(Model::find_cached(model_id));
    unwrap_or_error!(model.predict_proba(&features))
}

fn predict_model_joint(model_id: i64, features: Vec<f32>) -> Vec<f32> {
    let model = unwrap_or_error!(Model::find_cached(model_id));
    unwrap_or_error!(model.predict_joint(&features))
}

fn predict_model_batch(model_id: i64, features: Vec<f32>) -> Vec<f32> {
    let model = unwrap_or_error!(Model::find_cached(model_id));
    unwrap_or_error!(model.predict_batch(&features))
}

fn predict_model_row(model_id: i64, row: &mut Rows) -> f32 {
    let model = unwrap_or_error!(Model::find_cached(model_id));
    let snapshot = &model.snapshot;
    let numeric_encoded_features = model.numeric_encode_features(row);
    let features_width = snapshot.features_width();
    let mut processed = vec![0_f32; features_width];

    let feature_data =
        ndarray::ArrayView2::from_shape((1, features_width), &numeric_encoded_features).unwrap();

    Zip::from(feature_data.columns())
        .and(&snapshot.feature_positions)
        .for_each(|data, position| {
            let column = &snapshot.columns[position.column_position - 1];
            column.preprocess(&data, &mut processed, features_width, position.row_position);
        });
    unwrap_or_error!(model.predict(&processed))
}

// fn snapshot(
//     relation_name: &str,
//     y_column_name: &str,
//     test_size: Option<f32>,
//     test_sampling: Option<Sampling>,
//     preprocess: Option<serde_json::Value>,
// ) -> Vec<(String, String)> {
//     let test_size = test_size.unwrap_or(0.25);
//     let test_sampling = test_sampling.unwrap_or(Sampling::stratified);
//     let preprocess = preprocess.unwrap_or(serde_json::json!({}));
//
//     Snapshot::create(
//         relation_name,
//         Some(y_column_name[0].to_string()),
//         test_size,
//         test_sampling,
//         true,
//         &preprocess.to_string(),
//     );
//     vec![(relation_name.to_string(), y_column_name.to_string())]
// }

pub struct LoadDatasetScalar {}

impl VScalar for LoadDatasetScalar {
    unsafe fn func(
        func: &duckdb::vtab::FunctionInfo,
        input: &mut duckdb::core::DataChunkHandle,
        output: &mut duckdb::core::FlatVector,
    ) -> duckdb::Result<(), Box<dyn std::error::Error>> {
        let rows = input.len();
        let binding = input.flat_vector(0);
        let source_param = binding.as_slice::<duckdb_string_t>();
        let source_param: Vec<String> = source_param
            .to_vec()
            .iter()
            .take(rows)
            .map(String::from)
            .collect();

        let binding = input.flat_vector(1);
        let subset_param = binding.as_slice::<duckdb_string_t>();
        let subset_param: Vec<Option<String>> = subset_param
            .to_vec()
            .iter()
            .take(rows)
            .map(|s| String::from(s))
            .map(|s| if s.is_empty() { None } else { Some(s) })
            .collect();

        let binding = input.flat_vector(2);
        let limit_param = binding.as_slice::<i64>();
        let limit_param: Vec<Option<i64>> = limit_param
            .to_vec()
            .iter()
            .take(rows)
            .map(|&l| if l < 0 { None } else { Some(l) })
            .collect();

        let binding = input.flat_vector(3);
        let kwargs_param = binding.as_slice::<duckdb_string_t>();
        let kwargs_param: Vec<Option<serde_json::Value>> = kwargs_param
            .to_vec()
            .iter()
            .take(rows)
            .map(String::from)
            .map(|s| serde_json::from_str(&s).ok())
            .collect();

        let args = izip!(source_param, subset_param, limit_param, kwargs_param);
        let results: Vec<(String, i64)> = args
            .map(|(source, subset, limit, kwargs)| {
                let (name, rows) = load_dataset(&source, subset, limit, kwargs);
                (name, rows)
            })
            .collect();

        let final_results = results
            .iter()
            .map(|(_, rows)| vec![*rows])
            .flatten()
            .collect::<Vec<i64>>();
        let mut result_column = ListVector::from(output);
        result_column.set_child(final_results.iter().as_slice());

        for i in 0..rows {
            result_column.set_entry(i, i, 1)
        }
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Bigint),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        ])
    }

    fn return_type() -> LogicalTypeHandle {
        LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Integer))
    }
}

fn load_dataset(
    source: &str,
    subset: Option<String>,
    limit: Option<i64>,
    kwargs: Option<serde_json::Value>,
) -> (String, i64) {
    // let subset = subset.as_deref();
    let limit: Option<usize> = limit.map(|limit| limit.try_into().unwrap());
    let kwargs = kwargs.unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
    let (name, rows) = match source {
        "breast_cancer" => dataset::load_breast_cancer(limit),
        "diabetes" => dataset::load_diabetes(limit),
        "digits" => dataset::load_digits(limit),
        "iris" => dataset::load_iris(limit),
        "linnerud" => dataset::load_linnerud(limit),
        "wine" => dataset::load_wine(limit),
        _ => {
            let rows =
                match crate::bindings::transformers::load_dataset(source, subset, limit, &kwargs) {
                    Ok(rows) => rows,
                    Err(e) => panic!("{e}"),
                };
            (source.into(), rows as i64)
        }
    };
    (name, rows)
}

pub struct EmbedScalar {}

impl VScalar for EmbedScalar {
    unsafe fn func(
        func: &duckdb::vtab::FunctionInfo,
        input: &mut duckdb::core::DataChunkHandle,
        output: &mut duckdb::core::FlatVector,
    ) -> duckdb::Result<(), Box<dyn std::error::Error>> {
        let rows = input.len();
        let binding = input.flat_vector(0);
        let model_param = binding.as_slice::<duckdb_string_t>();
        let model_param: Vec<String> = model_param
            .to_vec()
            .iter()
            .take(rows)
            .map(String::from)
            .collect();
        let binding = input.flat_vector(2);
        let text_param = binding.as_slice::<duckdb_string_t>();
        let text_param: Vec<String> = text_param
            .to_vec()
            .iter()
            .take(rows)
            .map(String::from)
            .collect();
        let flat_vector = input.flat_vector(2);
        let kwargs_param = flat_vector.as_slice::<duckdb_string_t>();
        let kwargs_param: Vec<String> = kwargs_param
            .to_vec()
            .iter()
            .take(rows)
            .map(String::from)
            .collect();
        let args = izip!(model_param, text_param, kwargs_param);
        let results: Vec<Vec<f32>> = args
            .map(|(model, text, kwargs)| {
                embed(&model, &text, serde_json::from_str(&kwargs).unwrap())
            })
            .collect();
        let binding = Vec::from_iter(results.iter().map(|r| r.as_slice()).flatten().map(|v| *v));
        let final_results = binding.as_slice();
        let mut new_list_vector = ListVector::from(output);
        new_list_vector.set_child(final_results);
        for i in 0..rows {
            new_list_vector.set_entry(i, i * results[i].len(), results[i].len())
        }
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        ])
    }

    fn return_type() -> LogicalTypeHandle {
        LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Float))
    }
}

#[cfg(feature = "python")]
pub fn embed(transformer: &str, text: &str, kwargs: serde_json::Value) -> Vec<f32> {
    match crate::bindings::transformers::embed(transformer, vec![text], &kwargs) {
        Ok(output) => output.first().unwrap().to_vec(),
        Err(e) => panic!("{e}"),
    }
}

#[cfg(all(feature = "python", not(feature = "use_as_lib")))]
pub fn embed_batch(
    transformer: &str,
    inputs: Vec<&str>,
    kwargs: serde_json::Value,
) -> Vec<Vec<f32>> {
    match crate::bindings::transformers::embed(transformer, inputs, &kwargs) {
        Ok(output) => output,
        Err(e) => panic!("{e}"),
    }
}

#[cfg(all(feature = "python", not(feature = "use_as_lib")))]
pub fn rank(
    transformer: &str,
    query: &str,
    documents: Vec<&str>,
    kwargs: serde_json::Value,
) -> Vec<crate::bindings::transformers::RankResult> {
    match crate::bindings::transformers::rank(transformer, query, documents, &kwargs) {
        Ok(output) => output,
        Err(e) => panic!("{e}"),
    }
}

/// Clears the GPU cache.
///
/// # Arguments
///
/// * `memory_usage` - Optional parameter indicating the memory usage percentage (0.0 -> 1.0)
///
/// # Returns
///
/// Returns `true` if the GPU cache was successfully cleared, `false` otherwise.
/// # Example
///
/// ```postgresql
/// SELECT quackml.clear_gpu_cache(memory_usage => 0.5);
/// ```
pub fn clear_gpu_cache(memory_usage: Option<f32>) -> bool {
    match crate::bindings::transformers::clear_gpu_cache(memory_usage) {
        Ok(success) => success,
        Err(e) => panic!("{e}"),
    }
}

pub fn chunk(splitter: &str, text: &str, kwargs: serde_json::Value) -> Vec<(i64, String)> {
    let chunks = match crate::bindings::langchain::chunk(splitter, text, &kwargs) {
        Ok(chunks) => chunks,
        Err(e) => panic!("{e}"),
    };

    let chunks = chunks
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| (i as i64 + 1, chunk))
        .collect::<Vec<(i64, String)>>();

    chunks
}

#[allow(unused_variables)] // cache is maintained for api compatibility
pub fn transform_json(
    task: serde_json::Value,
    args: serde_json::Value,
    inputs: Vec<&str>,
) -> serde_json::Value {
    match crate::bindings::transformers::transform(&task, &args, inputs) {
        Ok(output) => output,
        Err(e) => panic!("{e}"),
    }
}

fn extract_text_from_json(json: &serde_json::Value) -> String {
    match json {
        serde_json::Value::Object(map) => {
            if let Some(text) = map.get("generated_text") {
                return text.as_str().unwrap_or("").to_string();
            }
            if let Some(text) = map.get("translation_text") {
                return text.as_str().unwrap_or("").to_string();
            }
            if let Some(text) = map.get("summary_text") {
                return text.as_str().unwrap_or("").to_string();
            }
            if let Some(text) = map.get("answer") {
                return text.as_str().unwrap_or("").to_string();
            }
            // Add more fields as needed for different task types
            "".to_string()
        }
        serde_json::Value::Array(arr) => arr
            .iter()
            .map(|v| extract_text_from_json(v))
            .collect::<Vec<String>>()
            .join(" "),
        _ => "".to_string(),
    }
}

pub struct TaskTransformScalar {}

impl VScalar for TaskTransformScalar {
    unsafe fn func(
        func: &duckdb::vtab::FunctionInfo,
        input: &mut DataChunkHandle,
        output: &mut duckdb::core::FlatVector,
    ) -> duckdb::Result<(), Box<dyn std::error::Error>> {
        let rows = input.len();
        let binding = input.flat_vector(0);
        let model_param = binding.as_slice::<duckdb_string_t>();
        let model_param: Vec<String> = model_param
            .to_vec()
            .iter()
            .take(rows)
            .map(String::from)
            .collect();
        let binding = input.flat_vector(1);
        let args_param = binding.as_slice::<duckdb_string_t>();
        let args_param: Vec<String> = args_param
            .to_vec()
            .iter()
            .take(rows)
            .map(String::from)
            .collect();
        let binding = input.list_vector(2);
        let inputs = binding.to_vec::<duckdb_string_t>(rows);
        let unwrapped_inputs: Vec<Vec<String>> = inputs
            .iter()
            .map(|input| input.iter().map(|s| String::from(&s.unwrap())).collect())
            .collect();
        let results: Vec<serde_json::Value> = izip!(model_param, args_param, unwrapped_inputs)
            .map(|(model, args, inputs)| {
                let model_value = match serde_json::from_str::<serde_json::Value>(&model) {
                    Ok(json_value) => transform_json(
                        json_value,
                        serde_json::Value::from_str(&args).unwrap(),
                        inputs.iter().map(|s| s.as_str()).collect(),
                    ),
                    Err(e) => transform_string(
                        model,
                        serde_json::Value::from_str(&args).unwrap(),
                        inputs.iter().map(|s| s.as_str()).collect(),
                    ),
                };
                model_value
            })
            .collect();
        println!("{:?}", results);
        let extracted_results: Vec<String> = results
            .iter()
            .map(|result| extract_text_from_json(result))
            .collect();
        let mut result_column = ListVector::from(output);
        let result_column_child = result_column.child(extracted_results.len());
        for (i, text) in extracted_results.iter().enumerate() {
            result_column_child.insert(i, text.as_str());
        }

        result_column.set_len(extracted_results.len());
        for i in 0..rows {
            result_column.set_entry(i, i * inputs[i].len(), inputs[i].len())
        }

        //     model_param,
        //     serde_json::Value::from_str(&args_param).unwrap(),
        //     unwrapped_inputs
        //         .to_vec()
        //         .iter()
        //         .map(|s| s.as_str())
        //         .collect(),
        // );
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Varchar)),
        ])
    }

    fn return_type() -> LogicalTypeHandle {
        LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Varchar))
    }
}
#[allow(unused_variables)] // cache is maintained for api compatibility
pub fn transform_string(
    task: String,
    args: serde_json::Value,
    inputs: Vec<&str>,
) -> serde_json::Value {
    let task_json = json!({ "task": task });
    match crate::bindings::transformers::transform(&task_json, &args, inputs) {
        Ok(output) => output,
        Err(e) => panic!("{e}"),
    }
}

// pub fn transform_conversational_json(
//     task: JsonB,
//     args: default!(JsonB, "'{}'"),
//     inputs: default!(Vec<JsonB>, "ARRAY[]::JSONB[]"),
//     cache: default!(bool, false),
// ) -> JsonB {
//     if !task.0["task"]
//         .as_str()
//         .is_some_and(|v| v == "conversational")
//     {
//         error!(
//             "ARRAY[]::JSONB inputs for transform should only be used with a conversational task"
//         );
//     }
//     if let Err(err) = crate::bindings::transformers::whitelist::verify_task(&task.0) {
//         error!("{err}");
//     }
//     match crate::bindings::transformers::transform(&task.0, &args.0, inputs) {
//         Ok(output) => JsonB(output),
//         Err(e) => error!("{e}"),
//     }
// }
//
// #[cfg(all(feature = "python", not(feature = "use_as_lib")))]
// #[pg_extern(immutable, parallel_safe, name = "transform")]
// #[allow(unused_variables)] // cache is maintained for api compatibility
// pub fn transform_conversational_string(
//     task: String,
//     args: default!(JsonB, "'{}'"),
//     inputs: default!(Vec<JsonB>, "ARRAY[]::JSONB[]"),
//     cache: default!(bool, false),
// ) -> JsonB {
//     if task != "conversational" {
//         error!(
//             "ARRAY[]::JSONB inputs for transform should only be used with a conversational task"
//         );
//     }
//     let task_json = json!({ "task": task });
//     if let Err(err) = crate::bindings::transformers::whitelist::verify_task(&task_json) {
//         error!("{err}");
//     }
//     match crate::bindings::transformers::transform(&task_json, &args.0, inputs) {
//         Ok(output) => JsonB(output),
//         Err(e) => error!("{e}"),
//     }
// }
//
// #[cfg(all(feature = "python", not(feature = "use_as_lib")))]
// #[pg_extern(immutable, parallel_safe, name = "transform_stream")]
// #[allow(unused_variables)] // cache is maintained for api compatibility
// pub fn transform_stream_json(
//     task: JsonB,
//     args: default!(JsonB, "'{}'"),
//     input: default!(&str, "''"),
//     cache: default!(bool, false),
// ) -> SetOfIterator<'static, JsonB> {
//     // We can unwrap this becuase if there is an error the current transaction is aborted in the map_err call
//     let python_iter =
//         crate::bindings::transformers::transform_stream_iterator(&task.0, &args.0, input)
//             .map_err(|e| error!("{e}"))
//             .unwrap();
//     SetOfIterator::new(python_iter)
// }
//
// #[cfg(all(feature = "python", not(feature = "use_as_lib")))]
// #[pg_extern(immutable, parallel_safe, name = "transform_stream")]
// #[allow(unused_variables)] // cache is maintained for api compatibility
// pub fn transform_stream_string(
//     task: String,
//     args: default!(JsonB, "'{}'"),
//     input: default!(&str, "''"),
//     cache: default!(bool, false),
// ) -> SetOfIterator<'static, JsonB> {
//     let task_json = json!({ "task": task });
//     // We can unwrap this becuase if there is an error the current transaction is aborted in the map_err call
//     let python_iter =
//         crate::bindings::transformers::transform_stream_iterator(&task_json, &args.0, input)
//             .map_err(|e| error!("{e}"))
//             .unwrap();
//     SetOfIterator::new(python_iter)
// }
//
// #[cfg(all(feature = "python", not(feature = "use_as_lib")))]
// #[pg_extern(immutable, parallel_safe, name = "transform_stream")]
// #[allow(unused_variables)] // cache is maintained for api compatibility
// pub fn transform_stream_conversational_json(
//     task: JsonB,
//     args: default!(JsonB, "'{}'"),
//     inputs: default!(Vec<JsonB>, "ARRAY[]::JSONB[]"),
//     cache: default!(bool, false),
// ) -> SetOfIterator<'static, JsonB> {
//     if !task.0["task"]
//         .as_str()
//         .is_some_and(|v| v == "conversational")
//     {
//         error!("ARRAY[]::JSONB inputs for transform_stream should only be used with a conversational task");
//     }
//     // We can unwrap this becuase if there is an error the current transaction is aborted in the map_err call
//     let python_iter =
//         crate::bindings::transformers::transform_stream_iterator(&task.0, &args.0, inputs)
//             .map_err(|e| error!("{e}"))
//             .unwrap();
//     SetOfIterator::new(python_iter)
// }
//
// #[cfg(all(feature = "python", not(feature = "use_as_lib")))]
// #[pg_extern(immutable, parallel_safe, name = "transform_stream")]
// #[allow(unused_variables)] // cache is maintained for api compatibility
// pub fn transform_stream_conversational_string(
//     task: String,
//     args: default!(JsonB, "'{}'"),
//     inputs: default!(Vec<JsonB>, "ARRAY[]::JSONB[]"),
//     cache: default!(bool, false),
// ) -> SetOfIterator<'static, JsonB> {
//     if task != "conversational" {
//         error!("ARRAY::JSONB inputs for transform_stream should only be used with a conversational task");
//     }
//     let task_json = json!({ "task": task });
//     // We can unwrap this becuase if there is an error the current transaction is aborted in the map_err call
//     let python_iter =
//         crate::bindings::transformers::transform_stream_iterator(&task_json, &args.0, inputs)
//             .map_err(|e| error!("{e}"))
//             .unwrap();
//     SetOfIterator::new(python_iter)
// }
pub struct GenerateScalar {}

impl VScalar for GenerateScalar {
    unsafe fn func(
        func: &duckdb::vtab::FunctionInfo,
        input: &mut DataChunkHandle,
        output: &mut duckdb::core::FlatVector,
    ) -> duckdb::Result<(), Box<dyn std::error::Error>> {
        let rows = input.len();
        let binding = input.flat_vector(0);
        let project_names = binding.as_slice::<duckdb_string_t>();
        let binding = input.flat_vector(1);
        let inputs = binding.as_slice::<duckdb_string_t>();
        let binding = input.flat_vector(2);
        let configs = binding.as_slice::<duckdb_string_t>();

        let results: Vec<String> = izip!(project_names, inputs, configs)
            .take(rows)
            .map(|(project_name, input, config)| {
                let project_name = String::from(project_name);
                let input = String::from(input);
                let config: serde_json::Value =
                    serde_json::from_str(&String::from(config)).unwrap();
                generate(&project_name, &input, config)
            })
            .collect();

        let mut result_column = ListVector::from(output);
        let result_column_child = result_column.child(results.len());
        for (i, text) in results.iter().enumerate() {
            result_column_child.insert(i, text.as_str());
        }

        result_column.set_len(results.len());
        for i in 0..rows {
            result_column.set_entry(i, i, 1);
        }

        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
            LogicalTypeHandle::from(LogicalTypeId::Varchar),
        ])
    }

    fn return_type() -> LogicalTypeHandle {
        LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Varchar))
    }
}

#[cfg(feature = "python")]
fn generate(project_name: &str, input: &str, config: serde_json::Value) -> String {
    generate_batch(project_name, vec![input], config)
        .first()
        .unwrap()
        .to_string()
}

#[cfg(feature = "python")]
fn generate_batch(project_name: &str, inputs: Vec<&str>, config: serde_json::Value) -> Vec<String> {
    use core::panic;

    match crate::bindings::transformers::generate(
        Project::get_deployed_model_id(project_name),
        inputs,
        config,
    ) {
        Ok(output) => output,
        Err(e) => panic!("{e}"),
    }
}

// Bind Data for Finetune Table Function
#[repr(C)]
pub struct FinetuneBindData {
    project_name: *mut c_char,
    task: *mut c_char,
    relation_name: *mut c_char,
    y_column_name: *mut c_char,
    model_name: *mut c_char, // New model parameter
    hyperparams: *mut c_char,
    test_size: *mut c_float,
    test_sampling: *mut c_char,
    automatic_deploy: *mut bool,
    materialize_snapshot: *mut bool,
}

impl Free for FinetuneBindData {
    fn free(&mut self) {
        unsafe {
            if !self.project_name.is_null() {
                drop(CString::from_raw(self.project_name));
            }
            if !self.task.is_null() {
                drop(CString::from_raw(self.task));
            }
            if !self.relation_name.is_null() {
                drop(CString::from_raw(self.relation_name));
            }
            if !self.y_column_name.is_null() {
                drop(CString::from_raw(self.y_column_name));
            }
            if !self.model_name.is_null() {
                drop(CString::from_raw(self.model_name));
            }
            if !self.hyperparams.is_null() {
                drop(CString::from_raw(self.hyperparams));
            }
            if !self.test_sampling.is_null() {
                drop(CString::from_raw(self.test_sampling));
            }
            if !self.automatic_deploy.is_null() {
                drop(Box::from_raw(self.automatic_deploy));
            }
            if !self.materialize_snapshot.is_null() {
                drop(Box::from_raw(self.materialize_snapshot));
            }
        }
    }
}

// Init Data for Finetune Table Function
#[repr(C)]
pub struct FinetuneInitData {
    done: bool,
}

impl Free for FinetuneInitData {
    fn free(&mut self) {}
}

pub struct FinetuneVTab;

impl VTab for FinetuneVTab {
    type InitData = FinetuneInitData;
    type BindData = FinetuneBindData;

    unsafe fn bind(
        bind: &duckdb::vtab::BindInfo,
        data: *mut Self::BindData,
    ) -> duckdb::Result<(), Box<dyn std::error::Error>> {
        // Define the output columns
        bind.add_result_column("status", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("task", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("algorithm", LogicalTypeHandle::from(LogicalTypeId::Varchar));
        bind.add_result_column("deployed", LogicalTypeHandle::from(LogicalTypeId::Boolean));

        // Extract parameters
        let project_name = bind.get_parameter(0).to_string();
        let task = bind
            .get_named_parameter("task")
            .map_or_else(|| "".to_string(), |v| v.to_string());
        let relation_name = bind
            .get_named_parameter("relation_name")
            .map_or_else(|| "".to_string(), |v| v.to_string());
        let y_column_name = bind
            .get_named_parameter("y_column_name")
            .map_or_else(|| "".to_string(), |v| v.to_string());
        let model_name = bind
            .get_named_parameter("model_name")
            .map_or_else(|| "".to_string(), |v| v.to_string());
        let hyperparams = bind
            .get_named_parameter("hyperparams")
            .map_or_else(|| "".to_string(), |v| v.to_string());
        let test_size = bind
            .get_named_parameter("test_size")
            .map_or_else(|| 0.0, |v| v.to_string().parse::<f32>().unwrap_or(0.0));
        let test_sampling = bind
            .get_named_parameter("test_sampling")
            .map_or_else(|| "".to_string(), |v| v.to_string());
        let automatic_deploy = bind
            .get_named_parameter("automatic_deploy")
            .map_or_else(|| false, |v| v.to_string().parse::<bool>().unwrap_or(false));
        let materialize_snapshot = bind
            .get_named_parameter("materialize_snapshot")
            .map_or_else(|| false, |v| v.to_string().parse::<bool>().unwrap_or(false));

        // Assign parameters to BindData
        (*data).project_name = CString::new(project_name).unwrap().into_raw();
        (*data).task = CString::new(task).unwrap().into_raw();
        (*data).relation_name = CString::new(relation_name).unwrap().into_raw();
        (*data).y_column_name = CString::new(y_column_name).unwrap().into_raw();
        (*data).model_name = CString::new(model_name).unwrap().into_raw();
        (*data).hyperparams = CString::new(hyperparams).unwrap().into_raw();
        (*data).test_sampling = CString::new(test_sampling).unwrap().into_raw();

        // Allocate heap memory for boolean parameters
        (*data).automatic_deploy = Box::into_raw(Box::new(automatic_deploy));
        (*data).materialize_snapshot = Box::into_raw(Box::new(materialize_snapshot));

        // Allocate heap memory for test_size
        (*data).test_size = Box::into_raw(Box::new(test_size));

        Ok(())
    }

    unsafe fn init(
        init: &InitInfo,
        data: *mut Self::InitData,
    ) -> duckdb::Result<(), Box<dyn std::error::Error>> {
        (*data).done = false;
        Ok(())
    }

    unsafe fn func(
        func: &FunctionInfo,
        output: &mut DataChunkHandle,
    ) -> duckdb::Result<(), Box<dyn std::error::Error>> {
        let init_info = func.get_init_data::<FinetuneInitData>();
        let bind_info = func.get_bind_data::<FinetuneBindData>();

        if (*init_info).done {
            // No more data to return
            output.set_len(0);
        } else {
            (*init_info).done = true;

            // Retrieve and clone the bound parameters
            let project_name = CString::from_raw((*bind_info).project_name);
            let task = CString::from_raw((*bind_info).task);
            let relation_name = CString::from_raw((*bind_info).relation_name);
            let y_column_name = CString::from_raw((*bind_info).y_column_name);
            let model_name = CString::from_raw((*bind_info).model_name);
            let hyperparams = CString::from_raw((*bind_info).hyperparams);
            let test_sampling = CString::from_raw((*bind_info).test_sampling);

            let automatic_deploy = *(*bind_info).automatic_deploy;
            let materialize_snapshot = *(*bind_info).materialize_snapshot;
            let test_size = *(*bind_info).test_size;

            // Reassign the raw pointers to prevent them from being freed
            (*bind_info).project_name = CString::into_raw(project_name.clone());
            (*bind_info).task = CString::into_raw(task.clone());
            (*bind_info).relation_name = CString::into_raw(relation_name.clone());
            (*bind_info).y_column_name = CString::into_raw(y_column_name.clone());
            (*bind_info).model_name = CString::into_raw(model_name.clone());
            (*bind_info).hyperparams = CString::into_raw(hyperparams.clone());
            (*bind_info).test_sampling = CString::into_raw(test_sampling.clone());

            // Convert C strings to Rust strings
            let project_name_str = project_name.to_str().unwrap();
            let task_opt = match task.to_str() {
                Ok("") => None,
                Ok(s) => Some(s),
                Err(_) => panic!("Failed to parse task string"),
            };
            let relation_name_opt = match relation_name.to_str() {
                Ok("") => None,
                Ok(s) => Some(s),
                Err(_) => panic!("Failed to parse relation_name string"),
            };
            let y_column_name_opt = match y_column_name.to_str() {
                Ok("") => None,
                Ok(s) => Some(vec![s.to_string()]),
                Err(_) => panic!("Failed to parse y_column_name string"),
            };
            let model_name_opt = match model_name.to_str() {
                Ok("") => None,
                Ok(s) => Some(s),
                Err(_) => panic!("Failed to parse model_name string"),
            };
            let hyperparams_map: Option<Hyperparams> = match hyperparams.to_str() {
                Ok("") => None,
                Ok(s) => Some(serde_json::from_str(s).unwrap()),
                Err(_) => panic!("Failed to parse hyperparams string"),
            };
            let test_sampling_enum = match test_sampling.to_str() {
                Ok("") => Sampling::stratified, // Default value
                Ok(s) => Sampling::from_str(s).unwrap_or(Sampling::stratified),
                Err(_) => panic!("Failed to parse test_sampling string"),
            };

            // Call the finetune function
            let result = tune(
                project_name_str,
                task_opt,
                relation_name_opt,
                y_column_name_opt,
                model_name_opt,
                &hyperparams_map.unwrap(),
                test_size,
                test_sampling_enum,
                None,
                materialize_snapshot,
            );

            // Populate the output columns
            let status_column = output.flat_vector(0);
            let task_column = output.flat_vector(1);
            let algorithm_column = output.flat_vector(2);
            let deployed_column: *mut bool = output.flat_vector(3).as_mut_ptr();
            let (status, task, algorithm, deployed) = result;

            // Insert the results
            status_column.insert(0, status.as_str());
            task_column.insert(0, task.as_str());
            algorithm_column.insert(0, algorithm.as_str());
            deployed_column.write(deployed);

            output.set_len(1);
        }

        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            LogicalTypeHandle::from(LogicalTypeId::Varchar), // project_name
        ])
    }

    fn named_parameters() -> Option<Vec<(String, LogicalTypeHandle)>> {
        Some(vec![
            (
                "task".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
            (
                "relation_name".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
            (
                "y_column_name".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
            (
                "model_name".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
            (
                "hyperparams".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
            (
                "test_size".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Float),
            ),
            (
                "test_sampling".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
            (
                "automatic_deploy".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Boolean),
            ),
            (
                "materialize_snapshot".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Boolean),
            ),
        ])
    }
}
//
// //// Define the TuneScalar Struct
// pub struct TuneScalar {}
//
// // Implement VScalar Trait for TuneScalar
// impl VScalar for TuneScalar {
//     unsafe fn func(
//         func: &duckdb::vtab::FunctionInfo,
//         input: &mut DataChunkHandle,
//         output: &mut duckdb::core::FlatVector,
//     ) -> duckdb::Result<(), Box<dyn std::error::Error>> {
//         // Number of rows to process
//         let rows = input.len();
//
//         // Extract input columns
//         // Column Order:
//         // 0: project_name (VARCHAR)
//         // 1: task (VARCHAR, nullable)
//         // 2: relation_name (VARCHAR, nullable)
//         // 3: _y_column_name (VARCHAR, nullable)
//         // 4: model_name (VARCHAR, nullable)
//         // 5: hyperparams (JSONB)
//         // 6: test_size (FLOAT)
//         // 7: test_sampling (VARCHAR)
//         // 8: automatic_deploy (BOOL, nullable)
//         // 9: materialize_snapshot (BOOL)
//
//         let project_name_col = input.flat_vector(0);
//         let binding = project_name_col.as_slice::<duckdb_string_t>();
//         let project_name = binding
//             .iter()
//             .take(rows)
//             .map(|s| String::from(s))
//             .collect::<Vec<String>>();
//
//         let task_col = input.flat_vector(1);
//         let binding = task_col.as_slice::<duckdb_string_t>();
//         let task = binding
//             .iter()
//             .take(rows)
//             .map(|s| String::from(s))
//             .collect::<Vec<String>>();
//
//         let relation_name_col = input.flat_vector(2);
//         let binding = relation_name_col.as_slice::<duckdb_string_t>();
//         let relation_name = binding
//             .iter()
//             .take(rows)
//             .map(|s| String::from(s))
//             .collect::<Vec<String>>();
//
//         let y_column_name_col = input.flat_vector(3);
//         let binding = y_column_name_col.as_slice::<duckdb_string_t>();
//         let y_column_name = binding
//             .iter()
//             .take(rows)
//             .map(|s| String::from(s))
//             .collect::<Vec<String>>();
//
//         let model_name_col = input.flat_vector(4);
//         let binding = model_name_col.as_slice::<duckdb_string_t>();
//         let model_name = binding
//             .iter()
//             .take(rows)
//             .map(|s| String::from(s))
//             .collect::<Vec<String>>();
//
//         let hyperparams_col = input.flat_vector(5);
//         let binding = hyperparams_col.as_slice::<duckdb_string_t>();
//         let hyperparams = binding
//             .iter()
//             .take(rows)
//             .map(|s| String::from(s).as_str())
//             .map(serde_json::from_str)
//             .map(|h| Hyperparams::from(h.unwrap()))
//             .collect::<Vec<Hyperparams>>();
//         let test_size_col = input.flat_vector(6);
//         let test_size: Vec<f32> = test_size_col.as_slice::<f32>().to_vec();
//
//         let test_sampling_col = input.flat_vector(7);
//         let test_sampling: Vec<String> = test_sampling_col
//             .as_slice::<duckdb_string_t>()
//             .iter()
//             .map(|s| String::from(s))
//             .collect::<Vec<String>>();
//
//         let automatic_deploy_col = input.flat_vector(8);
//         let automatic_deploy: Vec<Option<bool>> = automatic_deploy_col
//             .as_slice::<bool>()
//             .iter()
//             .map(|b| Some(*b))
//             .collect(); // Assuming non-nullable, else handle accordingly
//
//         let materialize_snapshot_col = input.flat_vector(9);
//         let materialize_snapshot: Vec<bool> = materialize_snapshot_col.as_slice::<bool>().to_vec();
//
//         // Prepare to collect outputs
//         let mut statuses = Vec::with_capacity(rows);
//         let mut tasks = Vec::with_capacity(rows);
//         let mut algorithms = Vec::with_capacity(rows);
//         let mut deployed_flags = Vec::with_capacity(rows);
//
//         // Iterate over each row and call the `tune` function
//         for i in 0..rows {
//             let project_name = &project_name[i];
//             let task_opt = &task[i];
//             let relation_name_opt = &relation_name[i];
//             let y_column_name_opt = &y_column_name[i];
//             let model_name_opt = &model_name[i];
//             let hyperparams = &hyperparams[i];
//             let test_size = test_size[i];
//             let test_sampling = &test_sampling[i];
//             let automatic_deploy_opt: Option<bool> = automatic_deploy[i];
//             let materialize_snapshot = materialize_snapshot[i];
//
//             // Convert Sampling string to Sampling enum
//             let test_sampling_enum =
//                 Sampling::from_str(test_sampling).unwrap_or(Sampling::stratified);
//
//             // Call the `tune` function
//             let result = tune(
//                 project_name,
//                 task_opt,
//                 Some(relation_name_opt),
//                 Some(y_column_name_opt),
//                 Some(model_name_opt),
//                 &hyperparams,
//                 test_size,
//                 test_sampling_enum,
//                 automatic_deploy_opt,
//                 materialize_snapshot,
//             );
//             let (status, task_str, algorithm, deployed) = result;
//
//             // Collect the results
//             statuses.push(status);
//             tasks.push(task_str);
//             algorithms.push(algorithm);
//             deployed_flags.push(deployed);
//
//
//         // Populate the output columns
//         // Output Schema: status (VARCHAR), task (VARCHAR), algorithm (VARCHAR), deployed (BOOL)
//
//         // Since DuckDB scalar functions return a single value, we'll serialize the output as JSONB
//         // Alternatively, consider implementing a Table-Valued Function (TVF) for multiple outputs
//
//         // Create a JSON array of results
//         let mut results_json = Vec::with_capacity(rows);
//         for i in 0..rows {
//             let result = json!({
//                 "status": statuses[i],
//                 "task": tasks[i],
//                 "algorithm": algorithms[i],
//                 "deployed": deployed_flags[i],
//             });
//             results_json.push(result);
//         }
//
//         // Serialize the JSON array
//         let serialized = serde_json::to_vec(&results_json)?;
//
//         // Set the output as JSONB
//         output.set_entry(0, &serialized);
//
//         Ok(())
//     }
//
//     fn parameters() -> Option<Vec<LogicalTypeHandle>> {
//         Some(vec![
//             LogicalTypeHandle::from(LogicalTypeId::Varchar), // project_name
//             LogicalTypeHandle::from(LogicalTypeId::Varchar), // task
//             LogicalTypeHandle::from(LogicalTypeId::Varchar), // relation_name
//             LogicalTypeHandle::from(LogicalTypeId::Varchar), // _y_column_name
//             LogicalTypeHandle::from(LogicalTypeId::Varchar), // model_name
//             LogicalTypeHandle::from(LogicalTypeId::Varchar), // hyperparams
//             LogicalTypeHandle::from(LogicalTypeId::Float),   // test_size
//             LogicalTypeHandle::from(LogicalTypeId::Varchar), // test_sampling
//             LogicalTypeHandle::from(LogicalTypeId::Boolean), // automatic_deploy
//             LogicalTypeHandle::from(LogicalTypeId::Boolean), // materialize_snapshot
//         ])
//     }
//
//     fn return_type() -> LogicalTypeHandle {
//         LogicalTypeHandle::from(LogicalTypeId::Varchar)
//     }
// }

#[cfg(feature = "python")]
fn tune(
    project_name: &str,
    task: Option<&str>,
    relation_name: Option<&str>,
    y_column_name: Option<Vec<String>>,
    model_name: Option<&str>,
    hyperparams: &Hyperparams,
    test_size: f32,
    test_sampling: Sampling,
    automatic_deploy: Option<bool>,
    materialize_snapshot: bool,
) -> (String, String, String, bool) {
    let task = task.map(Task::from_str).map(Result::unwrap);
    let preprocess = json!({}).to_string();
    let project = match Project::find_by_name(project_name) {
        Some(project) => project,
        None => Project::create(
            project_name,
            match task {
                Some(task) => task,
                None => panic!(
                    "Project `{}` does not exist. To create a new project, provide the task.",
                    project_name
                ),
            },
        ),
    };

    if task.is_some() && task.unwrap() != project.task {
        error!(
            "Project `{:?}` already exists with a different task: `{:?}`. Create a new project instead.",
            project.name, project.task
        );
    }

    let mut snapshot = match relation_name {
        None => {
            let snapshot = project.last_snapshot().expect(
                "You must pass a `relation_name` and `y_column_name` to snapshot the first time you train a model.",
            );

            info!("Using existing snapshot from {}", snapshot.snapshot_name(),);

            snapshot
        }

        Some(relation_name) => {
            info!(
                "Snapshotting table \"{}\", this may take a little while...",
                relation_name
            );

            let snapshot = Snapshot::create(
                relation_name,
                y_column_name,
                test_size,
                test_sampling,
                materialize_snapshot,
                preprocess.as_str(),
            );

            if materialize_snapshot {
                info!(
                    "Snapshot of table \"{}\" created and saved in {}",
                    relation_name,
                    snapshot.snapshot_name(),
                );
            }

            snapshot
        }
    };

    // algorithm will be transformers, stash the model_name in a hyperparam for v1 compatibility.
    let mut hyperparams = hyperparams.clone();
    hyperparams.insert(String::from("model_name"), json!(model_name));
    hyperparams.insert(String::from("project_name"), json!(project_name));
    let hyperparams = json!(hyperparams);

    // # Default repeatable random state when possible
    // let algorithm = Model.algorithm_from_name_and_task(algorithm, task);
    // if "random_state" in algorithm().get_params() and "random_state" not in hyperparams:
    //     hyperparams["random_state"] = 0
    let model = Model::finetune(&project, &mut snapshot, hyperparams).unwrap();
    let new_metrics: &serde_json::Value = &model.metrics.unwrap();
    let new_metrics = new_metrics.as_object().unwrap();

    let deployed_metrics = context::run(|conn| {
        conn.query_row(
            "
        SELECT models.metrics
        FROM quackml.models
        JOIN quackml.deployments
            ON deployments.model_id = models.id
        JOIN quackml.projects
            ON projects.id = deployments.project_id
        WHERE projects.name = $1
        ORDER by deployments.created_at DESC
        LIMIT 1;",
            params![project_name],
            |row| {
                let metrics: Option<String> = row.get(0).ok();
                Ok(metrics)
            },
        )
        .map_err(|e| anyhow::anyhow!("{e}"))
    });

    let mut deploy = true;
    match automatic_deploy {
        // Deploy only if metrics are better than previous model.
        Some(true) | None => {
            if let Ok(Some(deployed_metrics)) = deployed_metrics {
                let deployed_metrics: serde_json::Value =
                    serde_json::from_str(&deployed_metrics).unwrap();

                let deployed_value = deployed_metrics
                    .get(&project.task.default_target_metric())
                    .and_then(|value| value.as_f64())
                    .unwrap_or_default(); // Default to 0.0 if the key is not present or conversion fails

                // Get the value for the default target metric from new_metrics or provide a default value
                let new_value = new_metrics
                    .get(&project.task.default_target_metric())
                    .and_then(|value| value.as_f64())
                    .unwrap_or_default(); // Default to 0.0 if the key is not present or conversion fails

                if project.task.value_is_better(deployed_value, new_value) {
                    deploy = false;
                }
            }
        }

        Some(false) => deploy = false,
    };

    if deploy {
        project.deploy(model.id, Strategy::new_score);
    }

    (
        project.name,
        project.task.to_string(),
        model.algorithm.to_string(),
        deploy,
    )
}
//
// #[cfg(feature = "python")]
// #[pg_extern(name = "sklearn_f1_score")]
// pub fn sklearn_f1_score(ground_truth: Vec<f32>, y_hat: Vec<f32>) -> f32 {
//     unwrap_or_error!(crate::bindings::sklearn::f1(&ground_truth, &y_hat))
// }
//
// #[cfg(feature = "python")]
// #[pg_extern(name = "sklearn_r2_score")]
// pub fn sklearn_r2_score(ground_truth: Vec<f32>, y_hat: Vec<f32>) -> f32 {
//     unwrap_or_error!(crate::bindings::sklearn::r2(&ground_truth, &y_hat))
// }
//
// #[cfg(feature = "python")]
// #[pg_extern(name = "sklearn_regression_metrics")]
// pub fn sklearn_regression_metrics(ground_truth: Vec<f32>, y_hat: Vec<f32>) -> JsonB {
//     let metrics = unwrap_or_error!(crate::bindings::sklearn::regression_metrics(
//         &ground_truth,
//         &y_hat,
//     ));
//     JsonB(json!(metrics))
// }
//
// #[cfg(feature = "python")]
// #[pg_extern(name = "sklearn_classification_metrics")]
// pub fn sklearn_classification_metrics(
//     ground_truth: Vec<f32>,
//     y_hat: Vec<f32>,
//     num_classes: i64,
// ) -> JsonB {
//     let metrics = unwrap_or_error!(crate::bindings::sklearn::classification_metrics(
//         &ground_truth,
//         &y_hat,
//         num_classes as _
//     ));
//
//     JsonB(json!(metrics))
// }
//
// #[pg_extern]
// pub fn dump_all(path: &str) {
//     let p = std::path::Path::new(path).join("projects.csv");
//     Spi::run(&format!(
//         "COPY quackml.projects TO '{}' CSV HEADER",
//         p.to_str().unwrap()
//     ))
//     .unwrap();
//
//     let p = std::path::Path::new(path).join("snapshots.csv");
//     Spi::run(&format!(
//         "COPY quackml.snapshots TO '{}' CSV HEADER",
//         p.to_str().unwrap()
//     ))
//     .unwrap();
//
//     let p = std::path::Path::new(path).join("models.csv");
//     Spi::run(&format!(
//         "COPY quackml.models TO '{}' CSV HEADER",
//         p.to_str().unwrap()
//     ))
//     .unwrap();
//
//     let p = std::path::Path::new(path).join("files.csv");
//     Spi::run(&format!(
//         "COPY quackml.files TO '{}' CSV HEADER",
//         p.to_str().unwrap()
//     ))
//     .unwrap();
//
//     let p = std::path::Path::new(path).join("deployments.csv");
//     Spi::run(&format!(
//         "COPY quackml.deployments TO '{}' CSV HEADER",
//         p.to_str().unwrap()
//     ))
//     .unwrap();
// }
//
// pub fn load_all(path: &str) {
//     let p = std::path::Path::new(path).join("projects.csv");
//     Spi::run(&format!(
//         "COPY quackml.projects FROM '{}' CSV HEADER",
//         p.to_str().unwrap()
//     ))
//     .unwrap();
//
//     let p = std::path::Path::new(path).join("snapshots.csv");
//     Spi::run(&format!(
//         "COPY quackml.snapshots FROM '{}' CSV HEADER",
//         p.to_str().unwrap()
//     ))
//     .unwrap();
//
//     let p = std::path::Path::new(path).join("models.csv");
//     Spi::run(&format!(
//         "COPY quackml.models FROM '{}' CSV HEADER",
//         p.to_str().unwrap()
//     ))
//     .unwrap();
//
//     let p = std::path::Path::new(path).join("files.csv");
//     Spi::run(&format!(
//         "COPY quackml.files FROM '{}' CSV HEADER",
//         p.to_str().unwrap()
//     ))
//     .unwrap();
//
//     let p = std::path::Path::new(path).join("deployments.csv");
//     Spi::run(&format!(
//         "COPY quackml.deployments FROM '{}' CSV HEADER",
//         p.to_str().unwrap()
//     ))
//     .unwrap();
// }
//
// #[cfg(any(test, feature = "pg_test"))]
// #[pg_schema]
// mod tests {
//     use super::*;
//     use crate::orm::algorithm::Algorithm;
//     use crate::orm::dataset::{load_breast_cancer, load_diabetes, load_digits};
//     use crate::orm::runtime::Runtime;
//     use crate::orm::sampling::Sampling;
//     use crate::orm::Hyperparams;
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_intro_translation() {
//         let sql = "SELECT quackml.transform(
//             'translation_en_to_fr',
//             inputs => ARRAY[
//                 'Welcome to the future!',
//                 'Where have you been all this time?'
//             ]
//         ) AS french;";
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             {"translation_text": "Bienvenue à l'avenir!"},
//             {"translation_text": "Où êtes-vous allé tout ce temps?"}
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_intro_sentiment_analysis() {
//         let sql = "SELECT quackml.transform(
//             task   => 'text-classification',
//             inputs => ARRAY[
//                 'I love how amazingly simple ML has become!',
//                 'I hate doing mundane and thankless tasks. ☹️'
//             ]
//         ) AS positivity;";
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             {"label": "POSITIVE", "score": 0.9995759129524232},
//             {"label": "NEGATIVE", "score": 0.9903519749641418}
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_sentiment_analysis_specific_model() {
//         let sql = r#"SELECT quackml.transform(
//             inputs => ARRAY[
//                 'I love how amazingly simple ML has become!',
//                 'I hate doing mundane and thankless tasks. ☹️'
//             ],
//             task  => '{"task": "text-classification",
//                     "model": "finiteautomata/bertweet-base-sentiment-analysis"
//                     }'::JSONB
//         ) AS positivity;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             {"label": "POS", "score": 0.992932200431826},
//             {"label": "NEG", "score": 0.975599765777588}
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_sentiment_analysis_industry_specific_model() {
//         let sql = r#"SELECT quackml.transform(
//             inputs => ARRAY[
//                 'Stocks rallied and the British pound gained.',
//                 'Stocks making the biggest moves midday: Nvidia, Palantir and more'
//             ],
//             task => '{"task": "text-classification",
//                     "model": "ProsusAI/finbert"
//                     }'::JSONB
//         ) AS market_sentiment;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             {"label": "positive", "score": 0.8983612656593323},
//             {"label": "neutral", "score": 0.8062630891799927}
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_nli() {
//         let sql = r#"SELECT quackml.transform(
//             inputs => ARRAY[
//                 'A soccer game with multiple males playing. Some men are playing a sport.'
//             ],
//             task => '{"task": "text-classification",
//                     "model": "roberta-large-mnli"
//                     }'::JSONB
//         ) AS nli;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             {"label": "ENTAILMENT", "score": 0.98837411403656}
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_qnli() {
//         let sql = r#"SELECT quackml.transform(
//             inputs => ARRAY[
//                 'Where is the capital of France?, Paris is the capital of France.'
//             ],
//             task => '{"task": "text-classification",
//                     "model": "cross-encoder/qnli-electra-base"
//                     }'::JSONB
//         ) AS qnli;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             {"label": "LABEL_0", "score": 0.9978110194206238}
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_qqp() {
//         let sql = r#"SELECT quackml.transform(
//             inputs => ARRAY[
//                 'Which city is the capital of France?, Where is the capital of France?'
//             ],
//             task => '{"task": "text-classification",
//                     "model": "textattack/bert-base-uncased-QQP"
//                     }'::JSONB
//         ) AS qqp;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             {"label": "LABEL_0", "score": 0.9988721013069152}
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_grammatical_correctness() {
//         let sql = r#"SELECT quackml.transform(
//             inputs => ARRAY[
//                 'I will walk to home when I went through the bus.'
//             ],
//             task => '{"task": "text-classification",
//                     "model": "textattack/distilbert-base-uncased-CoLA"
//                     }'::JSONB
//         ) AS grammatical_correctness;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             {"label": "LABEL_1", "score": 0.9576480388641356}
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_zeroshot_classification() {
//         let sql = r#"SELECT quackml.transform(
//             inputs => ARRAY[
//                 'I have a problem with my iphone that needs to be resolved asap!!'
//             ],
//             task => '{
//                         "task": "zero-shot-classification",
//                         "model": "facebook/bart-large-mnli"
//                     }'::JSONB,
//             args => '{
//                         "candidate_labels": ["urgent", "not urgent", "phone", "tablet", "computer"]
//                     }'::JSONB
//         ) AS zero_shot;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             {
//                 "labels": ["urgent", "phone", "computer", "not urgent", "tablet"],
//                 "scores": [0.503635, 0.47879, 0.012600, 0.002655, 0.002308],
//                 "sequence": "I have a problem with my iphone that needs to be resolved asap!!"
//             }
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_token_classification_ner() {
//         let sql = r#"SELECT quackml.transform(
//             inputs => ARRAY[
//                 'I am Omar and I live in New York City.'
//             ],
//             task => 'token-classification'
//         ) as ner;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([[
//             {"end": 9,  "word": "Omar", "index": 3,  "score": 0.997110, "start": 5,  "entity": "I-PER"},
//             {"end": 27, "word": "New",  "index": 8,  "score": 0.999372, "start": 24, "entity": "I-LOC"},
//             {"end": 32, "word": "York", "index": 9,  "score": 0.999355, "start": 28, "entity": "I-LOC"},
//             {"end": 37, "word": "City", "index": 10, "score": 0.999431, "start": 33, "entity": "I-LOC"}
//         ]]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_token_classification_pos() {
//         let sql = r#"select quackml.transform(
//             inputs => array [
//             'I live in Amsterdam.'
//             ],
//             task => '{"task": "token-classification",
//                     "model": "vblagoje/bert-english-uncased-finetuned-pos"
//             }'::JSONB
//         ) as pos;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([[
//             {"end": 1,  "word": "i",         "index": 1, "score": 0.999, "start": 0,  "entity": "PRON"},
//             {"end": 6,  "word": "live",      "index": 2, "score": 0.998, "start": 2,  "entity": "VERB"},
//             {"end": 9,  "word": "in",        "index": 3, "score": 0.999, "start": 7,  "entity": "ADP"},
//             {"end": 19, "word": "amsterdam", "index": 4, "score": 0.998, "start": 10, "entity": "PROPN"},
//             {"end": 20, "word": ".",         "index": 5, "score": 0.999, "start": 19, "entity": "PUNCT"}
//         ]]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_translation() {
//         let sql = r#"select quackml.transform(
//             inputs => array[
//                         'How are you?'
//             ],
//             task => '{"task": "translation",
//                     "model": "Helsinki-NLP/opus-mt-en-fr"
//             }'::JSONB
//         );"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             {"translation_text": "Comment allez-vous ?"}
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_summarization() {
//         let sql = r#"select quackml.transform(
//             task => '{"task": "summarization",
//                     "model": "sshleifer/distilbart-cnn-12-6"
//             }'::JSONB,
//             inputs => array[
//             'Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018, in an area of more than 105 square kilometres (41 square miles). The City of Paris is the centre and seat of government of the region and province of Île-de-France, or Paris Region, which has an estimated population of 12,174,880, or about 18 percent of the population of France as of 2017.'
//             ]
//         );"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             {"summary_text": " Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018 . The city is the centre and seat of government of the region and province of Île-de-France, or Paris Region . Paris Region has an estimated 18 percent of the population of France as of 2017 ."}
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_summarization_min_max_length() {
//         let sql = r#"select quackml.transform(
//             task => '{"task": "summarization",
//                     "model": "sshleifer/distilbart-cnn-12-6"
//             }'::JSONB,
//             inputs => array[
//             'Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018, in an area of more than 105 square kilometres (41 square miles). The City of Paris is the centre and seat of government of the region and province of Île-de-France, or Paris Region, which has an estimated population of 12,174,880, or about 18 percent of the population of France as of 2017.'
//             ],
//             args => '{
//                     "min_length" : 20,
//                     "max_length" : 70
//             }'::JSONB
//         );"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             {"summary_text": " Paris is the capital and most populous city of France, with an estimated population of 2,175,601 residents as of 2018 . City of Paris is centre and seat of government of the region and province of Île-de-France, or Paris Region, which has an estimated 12,174,880, or about 18 percent"}
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_question_answering() {
//         let sql = r#"SELECT quackml.transform(
//             'question-answering',
//             inputs => ARRAY[
//                 '{
//                     "question": "Where do I live?",
//                     "context": "My name is Merve and I live in İstanbul."
//                 }'
//             ]
//         ) AS answer;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!({
//             "end"   :  39,
//             "score" :  0.9538117051124572,
//             "start" :  31,
//             "answer": "İstanbul"
//         });
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_text_generation() {
//         let sql = r#"SELECT quackml.transform(
//             task => 'text-generation',
//             inputs => ARRAY[
//                 'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone'
//             ]
//         ) AS answer;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             [
//                 {"generated_text": "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, and eight for the Dragon-lords in their halls of blood.\n\nEach of the guild-building systems is one-man"}
//             ]
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_text_generation_specific_model() {
//         let sql = r#"SELECT quackml.transform(
//             task => '{
//                 "task" : "text-generation",
//                 "model" : "gpt2-medium"
//             }'::JSONB,
//             inputs => ARRAY[
//                 'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone'
//             ]
//         ) AS answer;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             [{"generated_text": "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone.\n\nThis place has a deep connection to the lore of ancient Elven civilization. It is home to the most ancient of artifacts,"}]
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_text_generation_max_length() {
//         let sql = r#"SELECT quackml.transform(
//             task => '{
//                 "task" : "text-generation",
//                 "model" : "gpt2-medium"
//             }'::JSONB,
//             inputs => ARRAY[
//                 'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone'
//             ],
//             args => '{
//                     "max_length" : 200
//                 }'::JSONB
//         ) AS answer;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             [{"generated_text": "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, Three for the Dwarfs and the Elves, One for the Gnomes of the Mines, and Two for the Elves of Dross.\"\n\nHobbits: The Fellowship is the first book of J.R.R. Tolkien's story-cycle, and began with his second novel - The Two Towers - and ends in The Lord of the Rings.\n\n\nIt is a non-fiction novel, so there is no copyright claim on some parts of the story but the actual text of the book is copyrighted by author J.R.R. Tolkien.\n\n\nThe book has been classified into two types: fantasy novels and children's books\n\nHobbits: The Fellowship is the first book of J.R.R. Tolkien's story-cycle, and began with his second novel - The Two Towers - and ends in The Lord of the Rings.It"}]
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_text_generation_num_return_sequences() {
//         let sql = r#"SELECT quackml.transform(
//             task => '{
//                 "task" : "text-generation",
//                 "model" : "gpt2-medium"
//             }'::JSONB,
//             inputs => ARRAY[
//                 'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone'
//             ],
//             args => '{
//                     "num_return_sequences" : 3
//                 }'::JSONB
//         ) AS answer;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             [
//                 {"generated_text": "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, and Thirteen for the human-men in their hall of fire.\n\nAll of us, our families, and our people"},
//                 {"generated_text": "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, and the tenth for a King! As each of these has its own special story, so I have written them into the game."},
//                 {"generated_text": "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone… What's left in the end is your heart's desire after all!\n\nHans: (Trying to be brave)"}
//             ]
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_text_generation_beams_stopping() {
//         let sql = r#"SELECT quackml.transform(
//             task => '{
//                 "task" : "text-generation",
//                 "model" : "gpt2-medium"
//             }'::JSONB,
//             inputs => ARRAY[
//                 'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone'
//             ],
//             args => '{
//                     "num_beams" : 5,
//                     "early_stopping" : true
//                 }'::JSONB
//         ) AS answer;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([[
//             {"generated_text": "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, Nine for the Dwarves in their caverns of ice, Ten for the Elves in their caverns of fire, Eleven for the"}
//         ]]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_text_generation_temperature() {
//         let sql = r#"SELECT quackml.transform(
//             task => '{
//                 "task" : "text-generation",
//                 "model" : "gpt2-medium"
//             }'::JSONB,
//             inputs => ARRAY[
//                 'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone'
//             ],
//             args => '{
//                     "do_sample" : true,
//                     "temperature" : 0.9
//                 }'::JSONB
//         ) AS answer;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([[{"generated_text": "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, and Thirteen for the Giants and Men of S.A.\n\nThe First Seven-Year Time-Traveling Trilogy is"}]]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_text_generation_top_p() {
//         let sql = r#"SELECT quackml.transform(
//             task => '{
//                 "task" : "text-generation",
//                 "model" : "gpt2-medium"
//             }'::JSONB,
//             inputs => ARRAY[
//                 'Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone'
//             ],
//             args => '{
//                     "do_sample" : true,
//                     "top_p" : 0.8
//                 }'::JSONB
//         ) AS answer;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([[{"generated_text": "Three Rings for the Elven-kings under the sky, Seven for the Dwarf-lords in their halls of stone, Four for the Elves of the forests and fields, and Three for the Dwarfs and their warriors.\" ―Lord Rohan [src"}]]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_text_text_generation() {
//         let sql = r#"SELECT quackml.transform(
//             task => '{
//                 "task" : "text2text-generation"
//             }'::JSONB,
//             inputs => ARRAY[
//                 'translate from English to French: I''m very happy'
//             ]
//         ) AS answer;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             {"generated_text": "Je suis très heureux"}
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn readme_nlp_fill_mask() {
//         let sql = r#"SELECT quackml.transform(
//             task => '{
//                 "task" : "fill-mask"
//             }'::JSONB,
//             inputs => ARRAY[
//                 'Paris is the <mask> of France.'
//
//             ]
//         ) AS answer;"#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([
//             {"score": 0.679, "token": 812,   "sequence": "Paris is the capital of France.",    "token_str": " capital"},
//             {"score": 0.051, "token": 32357, "sequence": "Paris is the birthplace of France.", "token_str": " birthplace"},
//             {"score": 0.038, "token": 1144,  "sequence": "Paris is the heart of France.",      "token_str": " heart"},
//             {"score": 0.024, "token": 29778, "sequence": "Paris is the envy of France.",       "token_str": " envy"},
//             {"score": 0.022, "token": 1867,  "sequence": "Paris is the Capital of France.",    "token_str": " Capital"}
//         ]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     #[ignore = "requires model download"]
//     fn template() {
//         let sql = r#""#;
//         let got = Spi::get_one::<JsonB>(sql).unwrap().unwrap().0;
//         let want = serde_json::json!([]);
//         assert_eq!(got, want);
//     }
//
//     #[pg_test]
//     fn test_project_lifecycle() {
//         let project = Project::create("test", Task::regression);
//         assert!(project.id > 0);
//         assert!(Project::find(project.id).unwrap().id > 0);
//     }
//
//     #[pg_test]
//     fn test_snapshot_lifecycle() {
//         load_diabetes(Some(25));
//
//         let snapshot = Snapshot::create(
//             "quackml.diabetes",
//             Some(vec!["target".to_string()]),
//             0.5,
//             Sampling::last,
//             true,
//             JsonB(serde_json::Value::Object(Hyperparams::new())),
//         );
//         assert!(snapshot.id > 0);
//     }
//
//     #[pg_test]
//     fn test_not_fully_qualified_table() {
//         load_diabetes(Some(25));
//
//         let result = std::panic::catch_unwind(|| {
//             let _snapshot = Snapshot::create(
//                 "diabetes",
//                 Some(vec!["target".to_string()]),
//                 0.5,
//                 Sampling::last,
//                 true,
//                 JsonB(serde_json::Value::Object(Hyperparams::new())),
//             );
//         });
//
//         assert!(result.is_err());
//     }
//
//     #[pg_test]
//     fn test_train_regression() {
//         load_diabetes(None);
//
//         // Modify postgresql.conf and add shared_preload_libraries = 'quackml'
//         // to test deployments.
//         let setting =
//             Spi::get_one::<String>("select setting from pg_settings where name = 'data_directory'")
//                 .unwrap();
//
//         info!("Data directory: {}", setting.unwrap());
//
//         for runtime in [Runtime::python, Runtime::rust] {
//             let result: Vec<(String, String, String, bool)> = train(
//                 "Test project",
//                 Some(&Task::regression.to_string()),
//                 Some("quackml.diabetes"),
//                 Some("target"),
//                 Algorithm::linear,
//                 JsonB(serde_json::Value::Object(Hyperparams::new())),
//                 None,
//                 JsonB(serde_json::Value::Object(Hyperparams::new())),
//                 JsonB(serde_json::Value::Object(Hyperparams::new())),
//                 0.25,
//                 Sampling::last,
//                 Some(runtime),
//                 Some(true),
//                 false,
//                 JsonB(serde_json::Value::Object(Hyperparams::new())),
//             )
//             .collect();
//
//             assert_eq!(result.len(), 1);
//             assert_eq!(result[0].0, String::from("Test project"));
//             assert_eq!(result[0].1, String::from("regression"));
//             assert_eq!(result[0].2, String::from("linear"));
//             // assert_eq!(result[0].3, true);
//         }
//    }
//
//     #[pg_test]
//     fn test_train_multiclass_classification() {
//         load_digits(None);
//
//         // Modify postgresql.conf and add shared_preload_libraries = 'quackml'
//         // to test deployments.
//         let setting =
//             Spi::get_one::<String>("select setting from pg_settings where name = 'data_directory'")
//                 .unwrap();
//
//         info!("Data directory: {}", setting.unwrap());
//
//         for runtime in [Runtime::python, Runtime::rust] {
//             let result: Vec<(String, String, String, bool)> = train(
//                 "Test project 2",
//                 Some(&Task::classification.to_string()),
//                 Some("quackml.digits"),
//                 Some("target"),
//                 Algorithm::xgboost,
//                 JsonB(serde_json::Value::Object(Hyperparams::new())),
//                 None,
//                 JsonB(serde_json::Value::Object(Hyperparams::new())),
//                 JsonB(serde_json::Value::Object(Hyperparams::new())),
//                 0.25,
//                 Sampling::last,
//                 Some(runtime),
//                 Some(true),
//                 false,
//                 JsonB(serde_json::Value::Object(Hyperparams::new())),
//             )
//             .collect();
//
//             assert_eq!(result.len(), 1);
//             assert_eq!(result[0].0, String::from("Test project 2"));
//             assert_eq!(result[0].1, String::from("classification"));
//             assert_eq!(result[0].2, String::from("xgboost"));
//             // assert_eq!(result[0].3, true);
//         }
//     }
//
//     #[pg_test]
//     fn test_train_binary_classification() {
//         load_breast_cancer(None);
//
//         // Modify postgresql.conf and add shared_preload_libraries = 'quackml'
//         // to test deployments.
//         let setting =
//             Spi::get_one::<String>("select setting from pg_settings where name = 'data_directory'")
//                 .unwrap();
//
//         info!("Data directory: {}", setting.unwrap());
//
//         for runtime in [Runtime::python, Runtime::rust] {
//             let result: Vec<(String, String, String, bool)> = train(
//                 "Test project 3",
//                 Some(&Task::classification.to_string()),
//                 Some("quackml.breast_cancer"),
//                 Some("malignant"),
//                 Algorithm::xgboost,
//                 JsonB(serde_json::Value::Object(Hyperparams::new())),
//                 None,
//                 JsonB(serde_json::Value::Object(Hyperparams::new())),
//                 JsonB(serde_json::Value::Object(Hyperparams::new())),
//                 0.25,
//                 Sampling::last,
//                 Some(runtime),
//                 Some(true),
//                 true,
//                 JsonB(serde_json::Value::Object(Hyperparams::new())),
//             )
//             .collect();
//
//             assert_eq!(result.len(), 1);
//             assert_eq!(result[0].0, String::from("Test project 3"));
//             assert_eq!(result[0].1, String::from("classification"));
//             assert_eq!(result[0].2, String::from("xgboost"));
//             // assert_eq!(result[0].3, true);
//         }
//     }
//
//     #[pg_test]
//     fn test_dump_load() {
//         dump_all("/tmp");
//         load_all("/tmp");
//     }
// }
