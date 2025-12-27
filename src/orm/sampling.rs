use serde::Deserialize;

use super::snapshot::Column;

#[derive(Copy, Clone, Eq, PartialEq, Debug, Deserialize)]
#[allow(non_camel_case_types)]
pub enum Sampling {
    random,
    last,
    stratified,
}

impl std::str::FromStr for Sampling {
    type Err = ();

    fn from_str(input: &str) -> Result<Sampling, Self::Err> {
        match input {
            "random" => Ok(Sampling::random),
            "last" => Ok(Sampling::last),
            "stratified" => Ok(Sampling::stratified),
            _ => Err(()),
        }
    }
}

impl std::string::ToString for Sampling {
    fn to_string(&self) -> String {
        match *self {
            Sampling::random => "random".to_string(),
            Sampling::last => "last".to_string(),
            Sampling::stratified => "stratified".to_string(),
        }
    }
}

impl Sampling {
    // Implementing the sampling strategy in SQL
    // Effectively orders the table according to the train/test split
    // e.g. first N rows are train, last M rows are test
    // where M is configured by the user
    pub fn get_sql(&self, relation_name: &str, y_column_names: Vec<Column>) -> String {
        let col_string = y_column_names
            .iter()
            .map(|c| c.quoted_name())
            .collect::<Vec<String>>()
            .join(", ");
        match *self {
            Sampling::random => {
                format!("SELECT * FROM {relation_name} ORDER BY RANDOM()")
            }
            Sampling::last => {
                format!("SELECT * FROM {relation_name}")
            }
            Sampling::stratified => {
                format!(
                    "
                    SELECT {col_string}
                    FROM (
                        SELECT
                            *,
                        ROW_NUMBER() OVER(PARTITION BY {col_string} ORDER BY RANDOM()) AS rn
                        FROM {relation_name}
                    ) AS subquery
                    ORDER BY rn, RANDOM();
                "
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    const ALL_SAMPLINGS: &[(&str, Sampling)] = &[
        ("random", Sampling::random),
        ("last", Sampling::last),
        ("stratified", Sampling::stratified),
    ];

    #[test]
    fn test_from_str_all_samplings() {
        for (name, expected) in ALL_SAMPLINGS {
            let result = Sampling::from_str(name);
            assert!(result.is_ok(), "Failed to parse sampling: {}", name);
            assert_eq!(result.unwrap(), *expected, "Mismatch for sampling: {}", name);
        }
    }

    #[test]
    fn test_to_string_all_samplings() {
        for (expected_name, sampling) in ALL_SAMPLINGS {
            let result = sampling.to_string();
            assert_eq!(result, *expected_name, "Mismatch for sampling: {:?}", sampling);
        }
    }

    #[test]
    fn test_from_str_roundtrip() {
        for (name, _) in ALL_SAMPLINGS {
            let parsed = Sampling::from_str(name).unwrap();
            let stringified = parsed.to_string();
            assert_eq!(stringified, *name, "Roundtrip failed for: {}", name);
        }
    }

    #[test]
    fn test_from_str_invalid() {
        let invalid_names = &[
            "invalid",
            "RANDOM",
            "Random",
            "",
            "time_series",
        ];

        for name in invalid_names {
            let result = Sampling::from_str(name);
            assert!(result.is_err(), "Expected error for invalid sampling: '{}'", name);
        }
    }

    #[test]
    fn test_sampling_equality() {
        assert_eq!(Sampling::random, Sampling::random);
        assert_ne!(Sampling::random, Sampling::last);
        assert_ne!(Sampling::stratified, Sampling::last);
    }

    #[test]
    fn test_sampling_copy_clone() {
        let sampling = Sampling::stratified;
        let copied = sampling;
        let cloned = sampling.clone();
        assert_eq!(sampling, copied);
        assert_eq!(sampling, cloned);
    }

    #[test]
    fn test_sampling_debug() {
        let sampling = Sampling::random;
        let debug_str = format!("{:?}", sampling);
        assert_eq!(debug_str, "random");
    }
}
