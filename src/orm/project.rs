use std::{
    collections::HashMap,
    fmt::{Display, Error, Formatter},
    str::FromStr,
};

use chrono::{Date, DateTime, Utc};
use duckdb::{
    params,
    types::{EnumType, Value},
};
use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::context::DATABASE_CONTEXT;

use super::{Snapshot, Strategy, Task};

static PROJECT_ID_TO_DEPLOYED_MODEL_ID: Lazy<Mutex<HashMap<i64, i64>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));
static PROJECT_NAME_TO_PROJECT_ID: Lazy<Mutex<HashMap<String, i64>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Debug, Clone)]
pub struct Project {
    pub id: i64,
    pub name: String,
    pub task: Task,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Display for Project {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(
            f,
            "Project {{ id: {}, name: {}, task: {:?} }}",
            self.id, self.name, self.task
        )
    }
}

impl Project {
    pub fn get_deployed_model_id(project_name: &str) -> i64 {
        let mut projects = PROJECT_NAME_TO_PROJECT_ID.lock();
        let project_id = match projects.get(project_name) {
            Some(project_id) => *project_id,
            None => {
                let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
                let result = conn.query_row(
                    "SELECT deployments.project_id, deployments.model_id
                FROM quackml.deployments
                JOIN quackml.projects ON projects.id = deployments.project_id
                WHERE projects.name = $1
                ORDER BY deployments.created_at DESC
                LIMIT 1",
                    [project_name],
                    |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
                );
                let (project_id, model_id) = match result {
                    Ok(o) => o,
                    Err(_) => panic!(
                        "No deployed model exists for the project named: `{}`",
                        project_name
                    ),
                };

                projects.insert(project_name.to_string(), project_id);
                let mut projects = PROJECT_ID_TO_DEPLOYED_MODEL_ID.lock();
                if projects.len() == 1024 {
                    eprintln!("Active projects have exceeded capacity map, clearing caches.");
                    projects.clear();
                }
                projects.insert(project_id, model_id).unwrap();
                project_id
            }
        };
        *PROJECT_ID_TO_DEPLOYED_MODEL_ID
            .try_lock()
            .unwrap()
            .get(&project_id)
            .unwrap()
    }

    pub fn deploy(&self, model_id: i64, strategy: Strategy) {
        println!("Deploying model id: {:?}", model_id);
        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
        let _deployment_id = conn.execute(
            "INSERT INTO quackml.deployments (project_id, model_id, strategy) VALUES ($1, $2, $3::quackml.strategy)",
            params![&self.id, &model_id, &strategy.to_string()],
        ).unwrap();

        let mut projects = PROJECT_ID_TO_DEPLOYED_MODEL_ID.lock();

        if projects.len() == 1024 {
            eprintln!("Active projects has exceeded capacity map, clearing caches.");
            projects.clear();
        }
        projects.insert(self.id, model_id).unwrap();
    }

    pub fn find(id: i64) -> Option<Project> {
        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
        conn
        .query_row(
            "SELECT id, name, task::TEXT, created_at, updated_at FROM quackml.projects WHERE id = $1 LIMIT 1;",
            params![id],
            |row| {
                let project = Some(Project {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    task: Task::from_str(&row.get::<_, String>(2)?).unwrap(),
                    created_at: row.get::<_,duckdb::types::Value>(3).map(|v| match v {
                        duckdb::types::Value::Timestamp(ts, i) => DateTime::from_utc(
                            DateTime::from_timestamp(i, 0).unwrap().naive_utc(),
                            Utc,
                        ),
                        _ => panic!("Expected a timestamp"),

                    }).unwrap(),
                    updated_at: row.get::<_,duckdb::types::Value>(4).map(|v| match v {
                        duckdb::types::Value::Timestamp(ts, i) => DateTime::from_utc(
                            DateTime::from_timestamp(i, 0).unwrap().naive_utc(),
                            Utc,
                        ),
                        _ => panic!("Expected a timestamp"),

                    }).unwrap(),
                });
                Ok(project)
            },
        )
        .unwrap()
    }

    pub fn find_by_name(name: &str) -> Option<Project> {
        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
        conn
        .query_row(
            "SELECT id, name, task::TEXT, created_at, updated_at FROM quackml.projects WHERE name = $1 LIMIT 1;",
            params![name],
            |row| {
                let project = Project {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    task: Task::from_str(&row.get::<_, String>(2)?).unwrap(),
                    created_at: row.get::<_,duckdb::types::Value>(3).map(|v| match v {
                        duckdb::types::Value::Timestamp(ts, i) => DateTime::from_utc(
                            DateTime::from_timestamp(i, 0).unwrap().naive_utc(),
                            Utc,
                        ),
                        _ => panic!("Expected a timestamp"),

                    }).unwrap(),
                    updated_at: row.get::<_,duckdb::types::Value>(4).map(|v| match v {
                        duckdb::types::Value::Timestamp(ts, i) => DateTime::from_utc(
                            DateTime::from_timestamp(i, 0).unwrap().naive_utc(),
                            Utc,
                        ),
                        _ => panic!("Expected a timestamp"),

                    }).unwrap(),
                };
                Ok(project)
            },
        ).ok()
    }

    pub fn create(name: &str, task: Task) -> Project {
        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
        conn
        .query_row("INSERT INTO quackml.projects (name, task) VALUES ($1, $2) RETURNING id, name, task, created_at, updated_at;", params![name, task.to_string()],
    |row| {
        let project = Some(
            Project {
                id: row.get(0)?,
                name: row.get(1)?,
                task: row.get::<_, Value>(2).map(|v| match v {
                                duckdb::types::Value::Enum(s) => Task::from_str(&s).unwrap(),
                                _ => panic!("Expected a string"),
                            }).unwrap(),
                created_at: row.get(3).unwrap(),
                updated_at: row.get(4).unwrap(),
            }
        );
        Ok(project)
    }).unwrap().unwrap()
    }

    pub fn last_snapshot(&self) -> Option<Snapshot> {
        Snapshot::find_last_by_project_id(self.id)
    }
}
