//! Custom error types for quackML with helpful user-facing messages.
//!
//! This module provides structured errors that include:
//! - Clear, actionable error messages
//! - "Did you mean?" suggestions for typos
//! - Example SQL to help users fix issues

use std::fmt;

/// Main error type for quackML operations.
#[derive(Debug)]
pub enum QuackMLError {
    /// A required parameter was not provided.
    MissingRequired {
        parameter: &'static str,
        hint: String,
    },

    /// Project was not found and no task was specified to create one.
    ProjectNotFound {
        project_name: String,
        hint: String,
    },

    /// Project already exists with a different task type.
    ProjectTaskMismatch {
        project_name: String,
        existing_task: String,
        requested_task: String,
    },

    /// Unknown algorithm name provided.
    UnknownAlgorithm {
        input: String,
        suggestion: Option<String>,
        valid_options: Vec<String>,
    },

    /// Algorithm doesn't support the specified task.
    AlgorithmTaskMismatch {
        algorithm: String,
        task: String,
        valid_algorithms: Vec<String>,
    },

    /// Unknown task type provided.
    UnknownTask {
        input: String,
        suggestion: Option<String>,
        valid_options: Vec<String>,
    },

    /// Table or relation not found.
    TableNotFound {
        table: String,
        hint: String,
    },

    /// Column not found in table.
    ColumnNotFound {
        column: String,
        table: String,
        available_columns: Vec<String>,
    },

    /// Invalid hyperparameter for algorithm.
    InvalidHyperparameter {
        param: String,
        algorithm: String,
        suggestion: Option<String>,
        valid_params: Vec<String>,
    },

    /// Invalid hyperparameter value.
    InvalidHyperparameterValue {
        param: String,
        value: String,
        expected: String,
    },

    /// Invalid JSON format.
    InvalidJson {
        context: String,
        message: String,
    },

    /// Data validation error.
    DataValidation {
        message: String,
        hint: String,
    },

    /// Feature vector error.
    FeatureError {
        message: String,
    },

    /// Model not found or not deployed.
    ModelNotFound {
        project_name: String,
        hint: String,
    },

    /// Unsupported operation for task type.
    UnsupportedTaskOperation {
        operation: String,
        task: String,
        supported_tasks: Vec<String>,
    },

    /// Internal error (wraps other errors).
    Internal {
        message: String,
    },
}

impl QuackMLError {
    /// Calculate Levenshtein distance between two strings.
    fn levenshtein(a: &str, b: &str) -> usize {
        let a_len = a.len();
        let b_len = b.len();

        if a_len == 0 {
            return b_len;
        }
        if b_len == 0 {
            return a_len;
        }

        let mut prev_row: Vec<usize> = (0..=b_len).collect();
        let mut curr_row: Vec<usize> = vec![0; b_len + 1];

        for (i, a_char) in a.chars().enumerate() {
            curr_row[0] = i + 1;
            for (j, b_char) in b.chars().enumerate() {
                let cost = if a_char == b_char { 0 } else { 1 };
                curr_row[j + 1] = (curr_row[j] + 1)
                    .min(prev_row[j + 1] + 1)
                    .min(prev_row[j] + cost);
            }
            std::mem::swap(&mut prev_row, &mut curr_row);
        }

        prev_row[b_len]
    }

    /// Find the closest match from a list of valid options.
    pub fn suggest_similar(input: &str, valid: &[&str]) -> Option<String> {
        let input_lower = input.to_lowercase();
        valid
            .iter()
            .map(|v| (v, Self::levenshtein(&input_lower, &v.to_lowercase())))
            .filter(|(_, dist)| *dist <= 3) // Only suggest if within 3 edits
            .min_by_key(|(_, dist)| *dist)
            .map(|(v, _)| (*v).to_string())
    }

    /// Create an UnknownAlgorithm error with automatic suggestion.
    pub fn unknown_algorithm(input: &str, valid: &[&str]) -> Self {
        let suggestion = Self::suggest_similar(input, valid);
        QuackMLError::UnknownAlgorithm {
            input: input.to_string(),
            suggestion,
            valid_options: valid.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Create an UnknownTask error with automatic suggestion.
    pub fn unknown_task(input: &str, valid: &[&str]) -> Self {
        let suggestion = Self::suggest_similar(input, valid);
        QuackMLError::UnknownTask {
            input: input.to_string(),
            suggestion,
            valid_options: valid.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Create an InvalidHyperparameter error with automatic suggestion.
    pub fn invalid_hyperparam(param: &str, algorithm: &str, valid: &[&str]) -> Self {
        let suggestion = Self::suggest_similar(param, valid);
        QuackMLError::InvalidHyperparameter {
            param: param.to_string(),
            algorithm: algorithm.to_string(),
            suggestion,
            valid_params: valid.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Create a MissingRequired error for the 'task' parameter.
    pub fn missing_task() -> Self {
        QuackMLError::MissingRequired {
            parameter: "task",
            hint: "Specify the ML task type. Options: classification, regression, clustering, etc.\n\
                   Example: task => 'classification'\n\n\
                   See all tasks: SELECT * FROM list_tasks();".to_string(),
        }
    }

    /// Create a MissingRequired error for the 'relation_name' parameter.
    pub fn missing_relation_name() -> Self {
        QuackMLError::MissingRequired {
            parameter: "relation_name",
            hint: "Specify the table containing your training data.\n\
                   Example: relation_name => 'my_table'\n\n\
                   This should be an existing table or view in your database.".to_string(),
        }
    }

    /// Create a MissingRequired error for the 'y_column_name' parameter.
    pub fn missing_y_column_name() -> Self {
        QuackMLError::MissingRequired {
            parameter: "y_column_name",
            hint: "Specify the target column to predict.\n\
                   Example: y_column_name => 'target'\n\n\
                   This column should contain the labels/values you want to predict.".to_string(),
        }
    }

    /// Create a ProjectNotFound error.
    pub fn project_not_found(project_name: &str) -> Self {
        QuackMLError::ProjectNotFound {
            project_name: project_name.to_string(),
            hint: format!(
                "Project '{}' does not exist. To create a new project, specify a 'task' parameter.\n\n\
                 Example:\n\
                 SELECT * FROM train(\n\
                     '{}',\n\
                     task => 'classification',\n\
                     relation_name => 'my_table',\n\
                     y_column_name => 'target'\n\
                 );",
                project_name, project_name
            ),
        }
    }
}

impl fmt::Display for QuackMLError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuackMLError::MissingRequired { parameter, hint } => {
                write!(f, "Missing required parameter: '{}'\n\n{}", parameter, hint)
            }

            QuackMLError::ProjectNotFound { project_name, hint } => {
                write!(f, "Project '{}' not found.\n\n{}", project_name, hint)
            }

            QuackMLError::ProjectTaskMismatch {
                project_name,
                existing_task,
                requested_task,
            } => {
                write!(
                    f,
                    "Project '{}' already exists with task '{}', but you specified '{}'.\n\n\
                     To train with a different task, create a new project with a different name.",
                    project_name, existing_task, requested_task
                )
            }

            QuackMLError::UnknownAlgorithm {
                input,
                suggestion,
                valid_options,
            } => {
                let mut msg = format!("Unknown algorithm: '{}'", input);
                if let Some(s) = suggestion {
                    msg.push_str(&format!("\n\nDid you mean '{}'?", s));
                }
                let sample: Vec<_> = valid_options.iter().take(10).collect();
                msg.push_str(&format!(
                    "\n\nValid algorithms include: {}\n\nSee all: SELECT * FROM list_algorithms();",
                    sample.iter().map(|s| format!("'{}'", s)).collect::<Vec<_>>().join(", ")
                ));
                write!(f, "{}", msg)
            }

            QuackMLError::AlgorithmTaskMismatch {
                algorithm,
                task,
                valid_algorithms,
            } => {
                let sample: Vec<_> = valid_algorithms.iter().take(5).collect();
                write!(
                    f,
                    "Algorithm '{}' does not support task '{}'.\n\n\
                     Valid algorithms for {}: {}",
                    algorithm,
                    task,
                    task,
                    sample.iter().map(|s| format!("'{}'", s)).collect::<Vec<_>>().join(", ")
                )
            }

            QuackMLError::UnknownTask {
                input,
                suggestion,
                valid_options,
            } => {
                let mut msg = format!("Unknown task: '{}'", input);
                if let Some(s) = suggestion {
                    msg.push_str(&format!("\n\nDid you mean '{}'?", s));
                }
                msg.push_str(&format!(
                    "\n\nValid tasks: {}",
                    valid_options.iter().map(|s| format!("'{}'", s)).collect::<Vec<_>>().join(", ")
                ));
                write!(f, "{}", msg)
            }

            QuackMLError::TableNotFound { table, hint } => {
                write!(f, "Table '{}' not found.\n\n{}", table, hint)
            }

            QuackMLError::ColumnNotFound {
                column,
                table,
                available_columns,
            } => {
                let sample: Vec<_> = available_columns.iter().take(10).collect();
                write!(
                    f,
                    "Column '{}' not found in table '{}'.\n\nAvailable columns: {}",
                    column,
                    table,
                    sample.iter().map(|s| format!("'{}'", s)).collect::<Vec<_>>().join(", ")
                )
            }

            QuackMLError::InvalidHyperparameter {
                param,
                algorithm,
                suggestion,
                valid_params,
            } => {
                let mut msg = format!(
                    "Unknown hyperparameter '{}' for algorithm '{}'",
                    param, algorithm
                );
                if let Some(s) = suggestion {
                    msg.push_str(&format!("\n\nDid you mean '{}'?", s));
                }
                let sample: Vec<_> = valid_params.iter().take(10).collect();
                msg.push_str(&format!(
                    "\n\nValid parameters: {}",
                    sample.iter().map(|s| format!("'{}'", s)).collect::<Vec<_>>().join(", ")
                ));
                write!(f, "{}", msg)
            }

            QuackMLError::InvalidHyperparameterValue {
                param,
                value,
                expected,
            } => {
                write!(
                    f,
                    "Invalid value '{}' for hyperparameter '{}'.\n\nExpected: {}",
                    value, param, expected
                )
            }

            QuackMLError::InvalidJson { context, message } => {
                write!(f, "Invalid JSON in {}: {}", context, message)
            }

            QuackMLError::DataValidation { message, hint } => {
                write!(f, "Data validation error: {}\n\n{}", message, hint)
            }

            QuackMLError::FeatureError { message } => {
                write!(f, "Feature error: {}", message)
            }

            QuackMLError::ModelNotFound { project_name, hint } => {
                write!(
                    f,
                    "No deployed model found for project '{}'.\n\n{}",
                    project_name, hint
                )
            }

            QuackMLError::UnsupportedTaskOperation {
                operation,
                task,
                supported_tasks,
            } => {
                write!(
                    f,
                    "Operation '{}' is not supported for task '{}'.\n\nSupported tasks: {}",
                    operation,
                    task,
                    supported_tasks.join(", ")
                )
            }

            QuackMLError::Internal { message } => {
                write!(f, "Internal error: {}", message)
            }
        }
    }
}

impl std::error::Error for QuackMLError {}

// Allow conversion from QuackMLError to Box<dyn std::error::Error>
impl From<QuackMLError> for Box<dyn std::error::Error> {
    fn from(err: QuackMLError) -> Self {
        Box::new(err)
    }
}

// Allow conversion from QuackMLError to std::io::Error
impl From<QuackMLError> for std::io::Error {
    fn from(err: QuackMLError) -> Self {
        std::io::Error::new(std::io::ErrorKind::Other, err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein() {
        assert_eq!(QuackMLError::levenshtein("xgboost", "xgboost"), 0);
        assert_eq!(QuackMLError::levenshtein("xgbost", "xgboost"), 1);
        assert_eq!(QuackMLError::levenshtein("xgbostt", "xgboost"), 1);
        assert_eq!(QuackMLError::levenshtein("", "abc"), 3);
        assert_eq!(QuackMLError::levenshtein("abc", ""), 3);
    }

    #[test]
    fn test_suggest_similar() {
        let algorithms = &["xgboost", "lightgbm", "linear", "logistic", "svm"];

        // Exact match not suggested (distance 0)
        assert_eq!(
            QuackMLError::suggest_similar("xgboost", algorithms),
            Some("xgboost".to_string())
        );

        // Typo suggestion
        assert_eq!(
            QuackMLError::suggest_similar("xgbost", algorithms),
            Some("xgboost".to_string())
        );

        assert_eq!(
            QuackMLError::suggest_similar("lightgm", algorithms),
            Some("lightgbm".to_string())
        );

        // Too different - no suggestion
        assert_eq!(
            QuackMLError::suggest_similar("randomforest", algorithms),
            None
        );
    }

    #[test]
    fn test_missing_required_display() {
        let err = QuackMLError::missing_task();
        let msg = err.to_string();
        assert!(msg.contains("Missing required parameter: 'task'"));
        assert!(msg.contains("classification"));
    }

    #[test]
    fn test_unknown_algorithm_display() {
        let err = QuackMLError::unknown_algorithm(
            "xgbost",
            &["xgboost", "lightgbm", "linear"],
        );
        let msg = err.to_string();
        assert!(msg.contains("Unknown algorithm: 'xgbost'"));
        assert!(msg.contains("Did you mean 'xgboost'?"));
    }
}
