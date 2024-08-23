use crate::bindings::Bindings;
use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::Value;
use std::collections::HashMap;
use std::fmt::{Display, Error, Formatter};
use std::sync::Arc;

use super::{Algorithm, Project, Search, Snapshot, Status};

#[allow(clippy::type_complexity)]
static DEPLOYED_MODELS_BY_ID: Lazy<Mutex<HashMap<i64, Arc<Model>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Debug)]
pub struct Model {
    pub id: i64,
    pub project_id: i64,
    pub snapshot_id: i64,
    pub algorithm: Algorithm,
    pub hyperparams: Value,
    // pub runtime: Runtime,
    pub status: Status,
    pub metrics: Option<Value>,
    pub search: Option<Search>,
    pub search_params: Value,
    pub search_args: Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub project: Project,
    pub snapshot: Snapshot,
    pub bindings: Option<Box<dyn Bindings>>,
    pub num_classes: usize,
    pub num_features: usize,
}

impl Display for Model {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(
            f,
            "Model {{ id: {}, task: {:?}, algorithm: {:?} }}",
            self.id, self.project.task, self.algorithm
        )
    }
}
