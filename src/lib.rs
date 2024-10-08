mod api;
pub mod bindings;
mod context;
pub mod orm;

use std::{
    error::Error,
    ffi::{c_char, c_void, CString},
};

use crate::context::{init_database_context, DATABASE_CONTEXT};

static SCHEMA: &str = include_str!("sql/schema.sql");

use api::activate_venv;
use api::python_version;
use duckdb::{
    params,
    vtab::{BindInfo, Free, FunctionInfo, InitInfo, VTab},
    Connection, Result,
};
use duckdb_loadable_macros::duckdb_entrypoint_c_api;
use libduckdb_sys as ffi;
use orm::load_datasets;

// Exposes a extern C function named "quackml_ext_init" in the compiled dynamic library,
// the "entrypoint" that duckdb will use to load the extension.
#[duckdb_entrypoint_c_api(ext_name = "quack_ml", min_duckdb_version = "v0.0.1")]
pub fn quack_ml_init(conn: Connection) -> Result<(), Box<dyn Error>> {
    // TODO: This has to do with unsupported Capi version. Fix in libduckdb-sys
    // Define the struct to hold the connection
    init_database_context(&conn);
    run_schema_query()?;
    // load_datasets();
    let res = activate_venv("quackml-venv");
    println!("venv activation: {:?}", res);
    println!("Python version: {:?}", python_version());
    conn.execute("LOAD json;", params![])
        .expect("Expected to load json extension");
    conn.register_scalar_function::<api::LoadDatasetScalar>("load_dataset")?;
    conn.register_table_function::<api::TrainVTab>("train")?;
    conn.register_table_function::<api::FinetuneVTab>("finetune")?;
    conn.register_scalar_function::<api::PredictScalar>("predict")
        .expect("could not register scalar function");
    conn.register_scalar_function::<api::PredictStringScalar>("predict_text")
        .expect("could not register scalar function");
    conn.register_scalar_function::<api::PredictProbaScalar>("predict_proba")
        .expect("could not register scalar function");
    conn.register_scalar_function::<api::EmbedScalar>("embed")
        .expect("expected to register scalar function");
    conn.register_scalar_function::<api::TaskTransformScalar>("transform")
        .expect("could not register scalar function");
    conn.register_scalar_function::<api::GenerateScalar>("generate")
        .expect("could not register scalar function");
    Ok(())
}

fn run_schema_query() -> Result<()> {
    let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
    conn.execute_batch(SCHEMA)
}
