use anyhow::Result;
use duckdb::params;
use std::path::PathBuf;

use crate::context;

#[cfg(feature = "candle")]
pub mod candle;
#[cfg(feature = "python")]
pub mod hf_transformers;

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
