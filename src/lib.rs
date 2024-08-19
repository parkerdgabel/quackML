mod context;

use std::{
    error::Error,
    ffi::{c_char, c_void, CString},
};

use crate::context::{init_database_context, DATABASE_CONTEXT};

static SCHEMA: &str = include_str!("sql/schema.sql");

use duckdb::{
    vtab::{BindInfo, Free, FunctionInfo, InitInfo, VTab},
    Connection, Result,
};
use duckdb_loadable_macros::duckdb_entrypoint;
use libduckdb_sys as ffi;

// Exposes a extern C function named "quackml_ext_init" in the compiled dynamic library,
// the "entrypoint" that duckdb will use to load the extension.
#[duckdb_entrypoint]
pub fn quackml_ext_init(conn: Connection) -> Result<(), Box<dyn Error>> {
    // Define the struct to hold the connection
    init_database_context(conn);
    run_schema_query()?;
    Ok(())
}


fn run_schema_query() -> Result<()> {
    let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
    conn.execute_batch(SCHEMA)
}
