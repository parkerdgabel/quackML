//! Use virtualenv.

use anyhow::Result;
use duckdb::{params, Connection};
use log::*;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

// use crate::config::PGML_VENV;
use crate::create_pymodule;

create_pymodule!("/src/bindings/python/python.py");

pub fn activate_venv(venv: &Option<&str>) -> Result<bool> {
    Python::with_gil(|py| {
        let create_venv: Py<PyAny> = get_module!(PY_MODULE).getattr(py, "create_venv")?;
        let activate_venv: Py<PyAny> = get_module!(PY_MODULE).getattr(py, "activate_venv")?;
        let result: Py<PyAny> = match venv {
            Some(venv) => {
                activate_venv.call1(py, PyTuple::new(py, &[venv.to_string().into_py(py)]))?
            }
            None => create_venv.call0(py)?,
        };

        Ok(result.extract(py)?)
    })
}

pub fn activate() -> Result<bool> {
    activate_venv(&Option::None)
}

pub fn pip_freeze() -> Result<Vec<String>> {
    let packages = Python::with_gil(|py| -> Result<Vec<String>> {
        let freeze = get_module!(PY_MODULE).getattr(py, "freeze")?;
        let result = freeze.call0(py)?;

        Ok(result.extract(py)?)
    })?;

    Ok(packages)
}

pub fn validate_dependencies() -> Result<bool> {
    Python::with_gil(|py| {
        let sys = PyModule::import(py, "sys").unwrap();
        let executable: String = sys.getattr("executable").unwrap().extract().unwrap();
        let version: String = sys.getattr("version").unwrap().extract().unwrap();
        info!("Python version: {version}, executable: {}", executable);
        for module in ["xgboost", "lightgbm", "numpy", "sklearn"] {
            match py.import(module) {
                Ok(_) => (),
                Err(e) => {
                    panic!("The {module} package is missing. Install it with `sudo pip3 install {module}`\n{e}");
                }
            }
        }
    });

    let sklearn = package_version("sklearn")?;
    let xgboost = package_version("xgboost")?;
    let lightgbm = package_version("lightgbm")?;
    let numpy = package_version("numpy")?;

    info!("Scikit-learn {sklearn}, XGBoost {xgboost}, LightGBM {lightgbm}, NumPy {numpy}",);

    Ok(true)
}

pub fn version() -> Result<String> {
    Python::with_gil(|py| {
        let sys = PyModule::import(py, "sys").unwrap();
        let version: String = sys.getattr("version").unwrap().extract().unwrap();
        Ok(version)
    })
}

pub fn package_version(name: &str) -> Result<String> {
    let conn = Connection::open_in_memory()?;
    conn.execute(
        "CREATE TABLE packages (name VARCHAR, version VARCHAR)",
        params![],
    )?;

    Python::with_gil(|py| {
        let package = py.import(name)?;
        let version: String = package.getattr("__version__")?.extract()?;
        conn.execute(
            "INSERT INTO packages (name, version) VALUES (?, ?)",
            params![name, version],
        )?;
        Ok::<_, anyhow::Error>(())
    })?;

    let mut stmt = conn.prepare("SELECT version FROM packages WHERE name = ?")?;
    let mut rows = stmt.query(params![name])?;

    if let Some(row) = rows.next()? {
        Ok(row.get(0)?)
    } else {
        Err(anyhow::anyhow!("Package not found"))
    }
}
