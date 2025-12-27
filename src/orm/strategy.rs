use serde::Deserialize;

#[derive(Copy, Clone, Eq, PartialEq, Debug, Deserialize)]
#[allow(non_camel_case_types)]
pub enum Strategy {
    new_score,
    best_score,
    most_recent,
    rollback,
    specific,
}

impl std::str::FromStr for Strategy {
    type Err = ();

    fn from_str(input: &str) -> Result<Strategy, Self::Err> {
        match input {
            "new_score" => Ok(Strategy::new_score),
            "best_score" => Ok(Strategy::best_score),
            "most_recent" => Ok(Strategy::most_recent),
            "rollback" => Ok(Strategy::rollback),
            "specific" => Ok(Strategy::rollback),
            _ => Err(()),
        }
    }
}

impl std::string::ToString for Strategy {
    fn to_string(&self) -> String {
        match *self {
            Strategy::new_score => "new_score".to_string(),
            Strategy::best_score => "best_score".to_string(),
            Strategy::most_recent => "most_recent".to_string(),
            Strategy::rollback => "rollback".to_string(),
            Strategy::specific => "specific".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    #[test]
    fn test_from_str_all_strategies() {
        assert_eq!(Strategy::from_str("new_score").unwrap(), Strategy::new_score);
        assert_eq!(Strategy::from_str("best_score").unwrap(), Strategy::best_score);
        assert_eq!(Strategy::from_str("most_recent").unwrap(), Strategy::most_recent);
        assert_eq!(Strategy::from_str("rollback").unwrap(), Strategy::rollback);
        // Note: "specific" maps to rollback in the current implementation
        assert_eq!(Strategy::from_str("specific").unwrap(), Strategy::rollback);
    }

    #[test]
    fn test_to_string_all_strategies() {
        assert_eq!(Strategy::new_score.to_string(), "new_score");
        assert_eq!(Strategy::best_score.to_string(), "best_score");
        assert_eq!(Strategy::most_recent.to_string(), "most_recent");
        assert_eq!(Strategy::rollback.to_string(), "rollback");
        assert_eq!(Strategy::specific.to_string(), "specific");
    }

    #[test]
    fn test_from_str_invalid() {
        let invalid_names = &[
            "invalid",
            "NEW_SCORE",
            "Best_Score",
            "",
            "newest",
        ];

        for name in invalid_names {
            let result = Strategy::from_str(name);
            assert!(result.is_err(), "Expected error for invalid strategy: '{}'", name);
        }
    }

    #[test]
    fn test_strategy_equality() {
        assert_eq!(Strategy::new_score, Strategy::new_score);
        assert_ne!(Strategy::new_score, Strategy::best_score);
        assert_ne!(Strategy::rollback, Strategy::specific);
    }

    #[test]
    fn test_strategy_copy_clone() {
        let strategy = Strategy::best_score;
        let copied = strategy;
        let cloned = strategy.clone();
        assert_eq!(strategy, copied);
        assert_eq!(strategy, cloned);
    }

    #[test]
    fn test_strategy_debug() {
        let strategy = Strategy::most_recent;
        let debug_str = format!("{:?}", strategy);
        assert_eq!(debug_str, "most_recent");
    }
}
