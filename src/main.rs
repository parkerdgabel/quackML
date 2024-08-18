use std::error::Error;

extern crate duckdb;
extern crate duckdb_loadable_macros;
extern crate libduckdb_sys;

use duckdb::{
    core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId},
    vtab::{BindInfo, Free, FunctionInfo, InitInfo, VTab},
    Connection, Result,
};
use duckdb_loadable_macros::duckdb_entrypoint;
use libduckdb_sys as ffi;

// Exposes a extern C function named "quackml_ext_init" in the compiled dynamic library,
// the "entrypoint" that duckdb will use to load the extension.
#[duckdb_entrypoint]
pub fn quackml_ext_init(conn: Connection) -> Result<(), Box<dyn Error>> {

    Ok(())
}
