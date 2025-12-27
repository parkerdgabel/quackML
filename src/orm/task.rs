use serde::Deserialize;

#[derive(Copy, Clone, Eq, PartialEq, Debug, Deserialize)]
#[allow(non_camel_case_types)]
pub enum Task {
    regression,
    classification,
    decomposition,
    clustering,
    question_answering,
    summarization,
    translation,
    text_classification,
    text_generation,
    text2text,
    embedding,
    text_pair_classification,
    conversation,
}

impl Task {
    pub fn is_classification(&self) -> bool {
        match self {
            Task::classification
            | Task::text_classification
            | Task::text_pair_classification
            | Task::conversation => true,
            _ => false,
        }
    }

    pub fn is_regression(&self) -> bool {
        match self {
            Task::regression => true,
            _ => false,
        }
    }

    pub fn is_text_classification(&self) -> bool {
        match self {
            Task::text_classification | Task::text_pair_classification => true,
            _ => false,
        }
    }

    pub fn is_text_generation(&self) -> bool {
        match self {
            Task::text_generation | Task::text2text => true,
            _ => false,
        }
    }

    pub fn is_embedding(&self) -> bool {
        match self {
            Task::embedding => true,
            _ => false,
        }
    }

    pub fn is_conversation(&self) -> bool {
        match self {
            Task::conversation => true,
            _ => false,
        }
    }

    pub fn is_supervised(&self) -> bool {
        matches!(self, Task::regression | Task::classification)
    }

    pub fn default_target_metric(&self) -> String {
        match self {
            Task::regression => "r2",
            Task::classification => "f1",
            Task::decomposition => "cumulative_explained_variance",
            Task::clustering => "silhouette",
            Task::question_answering => "f1",
            Task::translation => "blue",
            Task::summarization => "rouge_ngram_f1",
            Task::text_classification => "f1",
            Task::text_generation => "perplexity",
            Task::text2text => "perplexity",
            Task::embedding => panic!("No default target metric for embedding task"),
            Task::text_pair_classification => "f1",
            Task::conversation => "bleu",
        }
        .to_string()
    }

    pub fn default_target_metric_positive(&self) -> bool {
        match self {
            Task::regression => true,
            Task::classification => true,
            Task::decomposition => true,
            Task::clustering => true,
            Task::question_answering => true,
            Task::translation => true,
            Task::summarization => true,
            Task::text_classification => true,
            Task::text_generation => false,
            Task::text2text => false,
            Task::embedding => panic!("No default target metric positive for embedding task"),
            Task::text_pair_classification => true,
            Task::conversation => true,
        }
    }

    pub fn value_is_better(&self, value: f64, other: f64) -> bool {
        if self.default_target_metric_positive() {
            value > other
        } else {
            value < other
        }
    }

    pub fn default_target_metric_sql_order(&self) -> String {
        let direction = if self.default_target_metric_positive() {
            "DESC"
        } else {
            "ASC"
        };
        format!(
            "ORDER BY models.metrics->>'{}' {} NULLS LAST",
            self.default_target_metric(),
            direction
        )
    }
}

impl std::str::FromStr for Task {
    type Err = ();

    fn from_str(input: &str) -> Result<Task, Self::Err> {
        match input {
            "regression" => Ok(Task::regression),
            "classification" => Ok(Task::classification),
            "decomposition" => Ok(Task::decomposition),
            "clustering" => Ok(Task::clustering),
            "question-answering" | "question_answering" => Ok(Task::question_answering),
            "summarization" => Ok(Task::summarization),
            "translation" => Ok(Task::translation),
            "text-classification" | "text_classification" => Ok(Task::text_classification),
            "text-generation" | "text_generation" => Ok(Task::text_generation),
            "text2text" => Ok(Task::text2text),
            "text-pair-classification" | "text_pair_classification" => {
                Ok(Task::text_pair_classification)
            }
            "conversation" => Ok(Task::conversation),
            _ => Err(()),
        }
    }
}

impl std::string::ToString for Task {
    fn to_string(&self) -> String {
        match *self {
            Task::regression => "regression".to_string(),
            Task::classification => "classification".to_string(),
            Task::decomposition => "decomposition".to_string(),
            Task::clustering => "clustering".to_string(),
            Task::question_answering => "question_answering".to_string(),
            Task::summarization => "summarization".to_string(),
            Task::translation => "translation".to_string(),
            Task::text_classification => "text_classification".to_string(),
            Task::text_generation => "text_generation".to_string(),
            Task::text2text => "text2text".to_string(),
            Task::embedding => "embedding".to_string(),
            Task::text_pair_classification => "text_pair_classification".to_string(),
            Task::conversation => "conversation".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    /// All task string names and their expected Task variants
    const ALL_TASKS: &[(&str, Task)] = &[
        ("regression", Task::regression),
        ("classification", Task::classification),
        ("decomposition", Task::decomposition),
        ("clustering", Task::clustering),
        ("question_answering", Task::question_answering),
        ("summarization", Task::summarization),
        ("translation", Task::translation),
        ("text_classification", Task::text_classification),
        ("text_generation", Task::text_generation),
        ("text2text", Task::text2text),
        ("text_pair_classification", Task::text_pair_classification),
        ("conversation", Task::conversation),
    ];

    /// Alternative names that should also parse correctly
    const ALTERNATIVE_NAMES: &[(&str, Task)] = &[
        ("question-answering", Task::question_answering),
        ("text-classification", Task::text_classification),
        ("text-generation", Task::text_generation),
        ("text-pair-classification", Task::text_pair_classification),
    ];

    #[test]
    fn test_from_str_all_tasks() {
        for (name, expected) in ALL_TASKS {
            let result = Task::from_str(name);
            assert!(result.is_ok(), "Failed to parse task: {}", name);
            assert_eq!(result.unwrap(), *expected, "Mismatch for task: {}", name);
        }
    }

    #[test]
    fn test_from_str_alternative_names() {
        for (name, expected) in ALTERNATIVE_NAMES {
            let result = Task::from_str(name);
            assert!(result.is_ok(), "Failed to parse alternative task name: {}", name);
            assert_eq!(result.unwrap(), *expected, "Mismatch for alternative task: {}", name);
        }
    }

    #[test]
    fn test_to_string_all_tasks() {
        for (expected_name, task) in ALL_TASKS {
            let result = task.to_string();
            assert_eq!(result, *expected_name, "Mismatch for task: {:?}", task);
        }
    }

    #[test]
    fn test_from_str_invalid() {
        let invalid_names = &[
            "invalid",
            "REGRESSION",
            "Classification",
            "",
            "   ",
            "embedding", // embedding is in enum but not in from_str
        ];

        for name in invalid_names {
            let result = Task::from_str(name);
            assert!(result.is_err(), "Expected error for invalid task: '{}'", name);
        }
    }

    #[test]
    fn test_is_classification() {
        let classification_tasks = &[
            Task::classification,
            Task::text_classification,
            Task::text_pair_classification,
            Task::conversation,
        ];

        let non_classification_tasks = &[
            Task::regression,
            Task::decomposition,
            Task::clustering,
            Task::question_answering,
            Task::summarization,
            Task::translation,
            Task::text_generation,
            Task::text2text,
            Task::embedding,
        ];

        for task in classification_tasks {
            assert!(task.is_classification(), "{:?} should be classification", task);
        }

        for task in non_classification_tasks {
            assert!(!task.is_classification(), "{:?} should NOT be classification", task);
        }
    }

    #[test]
    fn test_is_regression() {
        assert!(Task::regression.is_regression());

        let non_regression = &[
            Task::classification,
            Task::decomposition,
            Task::clustering,
            Task::text_classification,
            Task::embedding,
        ];

        for task in non_regression {
            assert!(!task.is_regression(), "{:?} should NOT be regression", task);
        }
    }

    #[test]
    fn test_is_text_classification() {
        assert!(Task::text_classification.is_text_classification());
        assert!(Task::text_pair_classification.is_text_classification());

        let non_text_classification = &[
            Task::classification,
            Task::regression,
            Task::text_generation,
            Task::conversation,
        ];

        for task in non_text_classification {
            assert!(!task.is_text_classification(), "{:?} should NOT be text_classification", task);
        }
    }

    #[test]
    fn test_is_text_generation() {
        assert!(Task::text_generation.is_text_generation());
        assert!(Task::text2text.is_text_generation());

        let non_text_generation = &[
            Task::classification,
            Task::text_classification,
            Task::conversation,
            Task::embedding,
        ];

        for task in non_text_generation {
            assert!(!task.is_text_generation(), "{:?} should NOT be text_generation", task);
        }
    }

    #[test]
    fn test_is_embedding() {
        assert!(Task::embedding.is_embedding());

        let non_embedding = &[
            Task::classification,
            Task::regression,
            Task::text_generation,
        ];

        for task in non_embedding {
            assert!(!task.is_embedding(), "{:?} should NOT be embedding", task);
        }
    }

    #[test]
    fn test_is_conversation() {
        assert!(Task::conversation.is_conversation());

        let non_conversation = &[
            Task::classification,
            Task::text_classification,
            Task::text_generation,
        ];

        for task in non_conversation {
            assert!(!task.is_conversation(), "{:?} should NOT be conversation", task);
        }
    }

    #[test]
    fn test_is_supervised() {
        assert!(Task::regression.is_supervised());
        assert!(Task::classification.is_supervised());

        let unsupervised = &[
            Task::clustering,
            Task::decomposition,
            Task::text_classification,
            Task::embedding,
        ];

        for task in unsupervised {
            assert!(!task.is_supervised(), "{:?} should NOT be supervised", task);
        }
    }

    #[test]
    fn test_default_target_metric() {
        assert_eq!(Task::regression.default_target_metric(), "r2");
        assert_eq!(Task::classification.default_target_metric(), "f1");
        assert_eq!(Task::decomposition.default_target_metric(), "cumulative_explained_variance");
        assert_eq!(Task::clustering.default_target_metric(), "silhouette");
        assert_eq!(Task::question_answering.default_target_metric(), "f1");
        assert_eq!(Task::translation.default_target_metric(), "blue");
        assert_eq!(Task::summarization.default_target_metric(), "rouge_ngram_f1");
        assert_eq!(Task::text_classification.default_target_metric(), "f1");
        assert_eq!(Task::text_generation.default_target_metric(), "perplexity");
        assert_eq!(Task::text2text.default_target_metric(), "perplexity");
        assert_eq!(Task::text_pair_classification.default_target_metric(), "f1");
        assert_eq!(Task::conversation.default_target_metric(), "bleu");
    }

    #[test]
    #[should_panic(expected = "No default target metric for embedding task")]
    fn test_default_target_metric_embedding_panics() {
        let _ = Task::embedding.default_target_metric();
    }

    #[test]
    fn test_default_target_metric_positive() {
        // Tasks where higher is better
        let positive_tasks = &[
            Task::regression,
            Task::classification,
            Task::decomposition,
            Task::clustering,
            Task::question_answering,
            Task::translation,
            Task::summarization,
            Task::text_classification,
            Task::text_pair_classification,
            Task::conversation,
        ];

        for task in positive_tasks {
            assert!(task.default_target_metric_positive(), "{:?} should have positive metric", task);
        }

        // Tasks where lower is better
        assert!(!Task::text_generation.default_target_metric_positive());
        assert!(!Task::text2text.default_target_metric_positive());
    }

    #[test]
    #[should_panic(expected = "No default target metric positive for embedding task")]
    fn test_default_target_metric_positive_embedding_panics() {
        let _ = Task::embedding.default_target_metric_positive();
    }

    #[test]
    fn test_value_is_better() {
        // For positive metrics (higher is better)
        assert!(Task::classification.value_is_better(0.9, 0.8));
        assert!(!Task::classification.value_is_better(0.8, 0.9));
        assert!(!Task::classification.value_is_better(0.8, 0.8));

        // For negative metrics (lower is better, like perplexity)
        assert!(Task::text_generation.value_is_better(10.0, 20.0));
        assert!(!Task::text_generation.value_is_better(20.0, 10.0));
    }

    #[test]
    fn test_default_target_metric_sql_order() {
        let order = Task::classification.default_target_metric_sql_order();
        assert!(order.contains("DESC"), "Classification should order DESC");
        assert!(order.contains("f1"), "Classification should order by f1");

        let order = Task::text_generation.default_target_metric_sql_order();
        assert!(order.contains("ASC"), "Text generation should order ASC");
        assert!(order.contains("perplexity"), "Text generation should order by perplexity");
    }

    #[test]
    fn test_task_equality() {
        assert_eq!(Task::classification, Task::classification);
        assert_ne!(Task::classification, Task::regression);
    }

    #[test]
    fn test_task_copy_clone() {
        let task = Task::regression;
        let copied = task;
        let cloned = task.clone();
        assert_eq!(task, copied);
        assert_eq!(task, cloned);
    }

    #[test]
    fn test_task_debug() {
        let task = Task::classification;
        let debug_str = format!("{:?}", task);
        assert_eq!(debug_str, "classification");
    }
}
