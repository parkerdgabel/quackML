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
            "INSERT INTO quackml.deployments (project_id, model_id, strategy) VALUES ($1, $2, $3::strategy)",
            params![&self.id, &model_id, &strategy.to_string()],
        ).unwrap();

        let mut projects = PROJECT_ID_TO_DEPLOYED_MODEL_ID.lock();

        if projects.len() == 1024 {
            eprintln!("Active projects has exceeded capacity map, clearing caches.");
            projects.clear();
        }
        match projects.insert(self.id, model_id) {
            Some(_) => println!("Updated project with id {}", self.id),
            None => println!("Inserted new project with id {}", model_id),
        }
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
                    created_at: row.get(3).unwrap(),
                    updated_at: row.get(4).unwrap()
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
                    created_at: row.get(3).unwrap(),
                    updated_at: row.get(4).unwrap(),
                };
                Ok(project)
            },
        ).ok()
    }

    pub fn create(name: &str, task: Task) -> Project {
        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
        conn
        .query_row("INSERT INTO quackml.projects (name, task) VALUES ($1, $2) RETURNING id, name, task::TEXT, created_at, updated_at;", params![name, task.to_string()],
    |row| {
        let project = Some(
            Project {
                id: row.get(0)?,
                name: row.get(1)?,
                task: row.get::<_, String>(2).map(|v| Task::from_str(&v).unwrap()).unwrap(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn create_test_project() -> Project {
        Project {
            id: 1,
            name: "test_project".to_string(),
            task: Task::classification,
            created_at: Utc.with_ymd_and_hms(2024, 1, 15, 10, 30, 0).unwrap(),
            updated_at: Utc.with_ymd_and_hms(2024, 1, 15, 12, 0, 0).unwrap(),
        }
    }

    #[test]
    fn test_project_display() {
        let project = create_test_project();
        let display = format!("{}", project);

        assert!(display.contains("Project"));
        assert!(display.contains("id: 1"));
        assert!(display.contains("name: test_project"));
        assert!(display.contains("task: classification"));
    }

    #[test]
    fn test_project_display_with_different_tasks() {
        let tasks = vec![
            Task::regression,
            Task::classification,
            Task::text_classification,
            Task::embedding,
        ];

        for task in tasks {
            let project = Project {
                id: 42,
                name: "ml_project".to_string(),
                task,
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            let display = format!("{}", project);
            assert!(display.contains("id: 42"));
            assert!(display.contains("ml_project"));
        }
    }

    #[test]
    fn test_project_clone() {
        let project = create_test_project();
        let cloned = project.clone();

        assert_eq!(project.id, cloned.id);
        assert_eq!(project.name, cloned.name);
        assert_eq!(project.task, cloned.task);
        assert_eq!(project.created_at, cloned.created_at);
        assert_eq!(project.updated_at, cloned.updated_at);
    }

    #[test]
    fn test_project_debug() {
        let project = create_test_project();
        let debug = format!("{:?}", project);

        assert!(debug.contains("Project"));
        assert!(debug.contains("id"));
        assert!(debug.contains("name"));
        assert!(debug.contains("task"));
    }

    #[test]
    fn test_project_fields_accessible() {
        let project = create_test_project();

        assert_eq!(project.id, 1);
        assert_eq!(project.name, "test_project");
        assert_eq!(project.task, Task::classification);
    }

    #[test]
    fn test_project_with_long_name() {
        let long_name = "a".repeat(1000);
        let project = Project {
            id: 1,
            name: long_name.clone(),
            task: Task::regression,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        assert_eq!(project.name, long_name);
        let display = format!("{}", project);
        assert!(display.contains(&long_name));
    }

    #[test]
    fn test_project_with_special_characters_in_name() {
        let special_name = "project-with_special.chars!@#$%";
        let project = Project {
            id: 1,
            name: special_name.to_string(),
            task: Task::classification,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        assert_eq!(project.name, special_name);
    }

    #[test]
    fn test_project_id_types() {
        // Test with large IDs
        let project = Project {
            id: i64::MAX,
            name: "max_id_project".to_string(),
            task: Task::regression,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        assert_eq!(project.id, i64::MAX);

        let project = Project {
            id: 0,
            name: "zero_id_project".to_string(),
            task: Task::regression,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        assert_eq!(project.id, 0);
    }
}
