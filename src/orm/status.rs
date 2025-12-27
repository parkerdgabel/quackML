use serde::Deserialize;

#[derive(Copy, Clone, Eq, PartialEq, Debug, Deserialize)]
#[allow(non_camel_case_types)]
pub enum Status {
    in_progress,
    successful,
    failed,
}

impl std::str::FromStr for Status {
    type Err = ();

    fn from_str(input: &str) -> Result<Status, Self::Err> {
        match input {
            "in_progress" => Ok(Status::in_progress),
            "successful" => Ok(Status::successful),
            "failed" => Ok(Status::failed),
            _ => Err(()),
        }
    }
}

impl std::string::ToString for Status {
    fn to_string(&self) -> String {
        match *self {
            Status::in_progress => "in_progress".to_string(),
            Status::successful => "successful".to_string(),
            Status::failed => "failed".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    const ALL_STATUSES: &[(&str, Status)] = &[
        ("in_progress", Status::in_progress),
        ("successful", Status::successful),
        ("failed", Status::failed),
    ];

    #[test]
    fn test_from_str_all_statuses() {
        for (name, expected) in ALL_STATUSES {
            let result = Status::from_str(name);
            assert!(result.is_ok(), "Failed to parse status: {}", name);
            assert_eq!(result.unwrap(), *expected, "Mismatch for status: {}", name);
        }
    }

    #[test]
    fn test_to_string_all_statuses() {
        for (expected_name, status) in ALL_STATUSES {
            let result = status.to_string();
            assert_eq!(result, *expected_name, "Mismatch for status: {:?}", status);
        }
    }

    #[test]
    fn test_from_str_roundtrip() {
        for (name, _) in ALL_STATUSES {
            let parsed = Status::from_str(name).unwrap();
            let stringified = parsed.to_string();
            assert_eq!(stringified, *name, "Roundtrip failed for: {}", name);
        }
    }

    #[test]
    fn test_from_str_invalid() {
        let invalid_names = &[
            "invalid",
            "IN_PROGRESS",
            "Successful",
            "",
            "pending",
            "running",
            "completed",
        ];

        for name in invalid_names {
            let result = Status::from_str(name);
            assert!(result.is_err(), "Expected error for invalid status: '{}'", name);
        }
    }

    #[test]
    fn test_status_equality() {
        assert_eq!(Status::in_progress, Status::in_progress);
        assert_ne!(Status::in_progress, Status::successful);
        assert_ne!(Status::successful, Status::failed);
    }

    #[test]
    fn test_status_copy_clone() {
        let status = Status::successful;
        let copied = status;
        let cloned = status.clone();
        assert_eq!(status, copied);
        assert_eq!(status, cloned);
    }

    #[test]
    fn test_status_debug() {
        let status = Status::failed;
        let debug_str = format!("{:?}", status);
        assert_eq!(debug_str, "failed");
    }
}
