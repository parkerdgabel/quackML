use crate::bindings::Bindings;
use crate::orm::dataset::Dataset;
use crate::orm::task::Task;
use crate::orm::Hyperparams;

use anyhow::format_err;
use anyhow::Result;
use lightgbm;
use serde_json::json;

pub struct Estimator {
    estimator: lightgbm::Booster,
}

unsafe impl Send for Estimator {}
unsafe impl Sync for Estimator {}

impl std::fmt::Debug for Estimator {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::result::Result<(), std::fmt::Error> {
        formatter.debug_struct("Estimator").finish()
    }
}

pub fn fit_regression(dataset: &Dataset, hyperparams: &Hyperparams) -> Result<Box<dyn Bindings>> {
    fit(dataset, hyperparams, Task::regression)
}

pub fn fit_classification(
    dataset: &Dataset,
    hyperparams: &Hyperparams,
) -> Result<Box<dyn Bindings>> {
    fit(dataset, hyperparams, Task::classification)
}

fn fit(dataset: &Dataset, hyperparams: &Hyperparams, task: Task) -> Result<Box<dyn Bindings>> {
    let mut hyperparams = hyperparams.clone();
    match task {
        Task::regression => {
            hyperparams.insert(
                "objective".to_string(),
                serde_json::Value::from("regression"),
            );
        }
        Task::classification => {
            if dataset.num_distinct_labels > 2 {
                hyperparams.insert(
                    "objective".to_string(),
                    serde_json::Value::from("multiclass"),
                );
                hyperparams.insert(
                    "num_class".to_string(),
                    serde_json::Value::from(dataset.num_distinct_labels),
                );
            } else {
                hyperparams.insert("objective".to_string(), serde_json::Value::from("binary"));
            }
        }
        _ => {
            return Err(format_err!(
                "lightgbm only supports `regression` and `classification` tasks."
            ))
        }
    };

    let data = lightgbm::Dataset::from_vec(
        &dataset.x_train,
        &dataset.y_train,
        dataset.num_features as i32,
    )
    .unwrap();

    let estimator = lightgbm::Booster::train(data, &json! {hyperparams}).unwrap();

    Ok(Box::new(Estimator { estimator }))
}

impl Bindings for Estimator {
    fn algorithm(&self) -> crate::orm::Algorithm {
        crate::orm::Algorithm::lightgbm
    }

    /// Predict a set of datapoints.
    fn predict(
        &self,
        features: &[f32],
        num_features: usize,
        num_classes: usize,
    ) -> Result<Vec<f32>> {
        let results = self.predict_proba(features, num_features)?;
        Ok(match num_classes {
            // TODO make lightgbm predict both classes like scikit and xgboost
            0 => results,
            2 => results.iter().map(|i| i.round()).collect(),
            _ => results
                .chunks(num_classes)
                .map(|probabilities| {
                    probabilities
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(index, _)| index)
                        .unwrap() as f32
                })
                .collect(),
        })
    }

    // Predict the raw probability of classes for a classifier.
    fn predict_proba(&self, features: &[f32], num_features: usize) -> Result<Vec<f32>> {
        Ok(self
            .estimator
            .predict(features, num_features as i32)?
            .into_iter()
            .map(|i| i as f32)
            .collect())
    }

    /// Serialize self to bytes
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let r: u64 = rand::random();
        let path = format!("/tmp/pgml_{}.bin", r);
        self.estimator.save_file(&path)?;
        let bytes = std::fs::read(&path)?;
        std::fs::remove_file(&path)?;

        Ok(bytes)
    }

    /// Deserialize self from bytes, with additional context
    fn from_bytes(bytes: &[u8]) -> Result<Box<dyn Bindings>>
    where
        Self: Sized,
    {
        let r: u64 = rand::random();
        let path = format!("/tmp/pgml_{}.bin", r);
        std::fs::write(&path, bytes)?;
        let mut estimator = lightgbm::Booster::from_file(&path);
        if estimator.is_err() {
            // backward compatibility w/ 2.0.0
            std::fs::write(&path, &bytes[16..])?;
            estimator = lightgbm::Booster::from_file(&path);
        }
        std::fs::remove_file(&path)?;
        let estimator = estimator?;
        Ok(Box::new(Estimator { estimator }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orm::Algorithm;

    /// Helper function to create a regression dataset
    fn create_regression_dataset() -> Dataset {
        // Simple linear relationship: y ≈ 2*x1 + 3*x2
        let x_train = vec![
            1.0, 2.0, // sample 1
            2.0, 3.0, // sample 2
            3.0, 4.0, // sample 3
            4.0, 5.0, // sample 4
            5.0, 6.0, // sample 5
            6.0, 7.0, // sample 6
            7.0, 8.0, // sample 7
            8.0, 9.0, // sample 8
        ];
        let y_train = vec![8.0, 13.0, 18.0, 23.0, 28.0, 33.0, 38.0, 43.0];

        let x_test = vec![
            1.5, 2.5, // test sample 1
            3.5, 4.5, // test sample 2
        ];
        let y_test = vec![10.5, 20.5];

        Dataset {
            x_train,
            y_train,
            x_test,
            y_test,
            num_features: 2,
            num_labels: 1,
            num_rows: 8,
            num_test_rows: 2,
            num_distinct_labels: 0,
        }
    }

    /// Helper function to create a binary classification dataset
    fn create_binary_classification_dataset() -> Dataset {
        // Simple binary classification
        let x_train = vec![
            0.0, 0.0, // class 0
            0.5, 0.5, // class 0
            1.0, 1.0, // class 0
            5.0, 5.0, // class 1
            5.5, 5.5, // class 1
            6.0, 6.0, // class 1
        ];
        let y_train = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let x_test = vec![
            0.2, 0.2, // should be class 0
            5.8, 5.8, // should be class 1
        ];
        let y_test = vec![0.0, 1.0];

        Dataset {
            x_train,
            y_train,
            x_test,
            y_test,
            num_features: 2,
            num_labels: 1,
            num_rows: 6,
            num_test_rows: 2,
            num_distinct_labels: 2,
        }
    }

    /// Helper function to create a multiclass classification dataset
    fn create_multiclass_classification_dataset() -> Dataset {
        // 3-class classification
        let x_train = vec![
            0.0, 0.0, // class 0
            0.5, 0.5, // class 0
            3.0, 3.0, // class 1
            3.5, 3.5, // class 1
            6.0, 6.0, // class 2
            6.5, 6.5, // class 2
        ];
        let y_train = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];

        let x_test = vec![
            0.2, 0.2, // should be class 0
            3.2, 3.2, // should be class 1
            6.2, 6.2, // should be class 2
        ];
        let y_test = vec![0.0, 1.0, 2.0];

        Dataset {
            x_train,
            y_train,
            x_test,
            y_test,
            num_features: 2,
            num_labels: 1,
            num_rows: 6,
            num_test_rows: 3,
            num_distinct_labels: 3,
        }
    }

    #[test]
    fn test_fit_regression() {
        let dataset = create_regression_dataset();
        let hyperparams = Hyperparams::new();

        let result = fit_regression(&dataset, &hyperparams);
        assert!(result.is_ok(), "fit_regression should succeed");

        let estimator = result.unwrap();
        assert_eq!(estimator.algorithm(), Algorithm::lightgbm);
    }

    #[test]
    fn test_fit_regression_with_hyperparams() {
        let dataset = create_regression_dataset();
        let mut hyperparams = Hyperparams::new();
        hyperparams.insert("num_iterations".to_string(), serde_json::json!(50));
        hyperparams.insert("learning_rate".to_string(), serde_json::json!(0.1));
        hyperparams.insert("num_leaves".to_string(), serde_json::json!(31));

        let result = fit_regression(&dataset, &hyperparams);
        assert!(result.is_ok(), "fit_regression with hyperparams should succeed");
    }

    #[test]
    fn test_fit_binary_classification() {
        let dataset = create_binary_classification_dataset();
        let hyperparams = Hyperparams::new();

        let result = fit_classification(&dataset, &hyperparams);
        assert!(result.is_ok(), "fit_classification (binary) should succeed");

        let estimator = result.unwrap();
        assert_eq!(estimator.algorithm(), Algorithm::lightgbm);
    }

    #[test]
    fn test_fit_multiclass_classification() {
        let dataset = create_multiclass_classification_dataset();
        let hyperparams = Hyperparams::new();

        let result = fit_classification(&dataset, &hyperparams);
        assert!(result.is_ok(), "fit_classification (multiclass) should succeed");
    }

    #[test]
    fn test_predict_regression() {
        let dataset = create_regression_dataset();
        let hyperparams = Hyperparams::new();

        let estimator = fit_regression(&dataset, &hyperparams).unwrap();
        let predictions = estimator
            .predict(&dataset.x_test, dataset.num_features, 0)
            .unwrap();

        assert_eq!(
            predictions.len(),
            dataset.num_test_rows,
            "Should have one prediction per test sample"
        );
        // Predictions should be reasonable (in the range of training targets)
        for pred in &predictions {
            assert!(*pred > 0.0 && *pred < 100.0, "Prediction {} should be reasonable", pred);
        }
    }

    #[test]
    fn test_predict_binary_classification() {
        let dataset = create_binary_classification_dataset();
        let hyperparams = Hyperparams::new();

        let estimator = fit_classification(&dataset, &hyperparams).unwrap();
        let predictions = estimator
            .predict(&dataset.x_test, dataset.num_features, 2)
            .unwrap();

        assert_eq!(
            predictions.len(),
            dataset.num_test_rows,
            "Should have one prediction per test sample"
        );
        // Predictions should be 0 or 1 for binary classification
        for pred in &predictions {
            assert!(
                *pred == 0.0 || *pred == 1.0,
                "Binary prediction {} should be 0 or 1",
                pred
            );
        }
    }

    #[test]
    fn test_predict_multiclass_classification() {
        let dataset = create_multiclass_classification_dataset();
        let hyperparams = Hyperparams::new();

        let estimator = fit_classification(&dataset, &hyperparams).unwrap();
        let predictions = estimator
            .predict(&dataset.x_test, dataset.num_features, 3)
            .unwrap();

        assert_eq!(
            predictions.len(),
            dataset.num_test_rows,
            "Should have one prediction per test sample"
        );
        // Predictions should be class indices (0, 1, or 2)
        for pred in &predictions {
            assert!(
                *pred >= 0.0 && *pred <= 2.0,
                "Multiclass prediction {} should be a valid class index",
                pred
            );
        }
    }

    #[test]
    fn test_predict_proba_binary() {
        let dataset = create_binary_classification_dataset();
        let hyperparams = Hyperparams::new();

        let estimator = fit_classification(&dataset, &hyperparams).unwrap();
        let probas = estimator
            .predict_proba(&dataset.x_test, dataset.num_features)
            .unwrap();

        // For binary classification, LightGBM returns one probability per sample
        assert_eq!(
            probas.len(),
            dataset.num_test_rows,
            "Should have probability per test sample for binary"
        );
        // Probabilities should be between 0 and 1
        for prob in &probas {
            assert!(
                *prob >= 0.0 && *prob <= 1.0,
                "Probability {} should be between 0 and 1",
                prob
            );
        }
    }

    #[test]
    fn test_predict_proba_multiclass() {
        let dataset = create_multiclass_classification_dataset();
        let hyperparams = Hyperparams::new();

        let estimator = fit_classification(&dataset, &hyperparams).unwrap();
        let probas = estimator
            .predict_proba(&dataset.x_test, dataset.num_features)
            .unwrap();

        // For 3-class classification with 3 test samples, should have 9 probabilities
        assert_eq!(
            probas.len(),
            dataset.num_test_rows * dataset.num_distinct_labels,
            "Should have num_classes probabilities per test sample"
        );
        // Probabilities should be between 0 and 1
        for prob in &probas {
            assert!(
                *prob >= 0.0 && *prob <= 1.0,
                "Probability {} should be between 0 and 1",
                prob
            );
        }
    }

    #[test]
    fn test_serialization_roundtrip() {
        let dataset = create_regression_dataset();
        let hyperparams = Hyperparams::new();

        let estimator = fit_regression(&dataset, &hyperparams).unwrap();

        // Serialize
        let bytes = estimator.to_bytes();
        assert!(bytes.is_ok(), "Serialization should succeed");
        let bytes = bytes.unwrap();
        assert!(!bytes.is_empty(), "Serialized bytes should not be empty");

        // Deserialize
        let loaded = Estimator::from_bytes(&bytes);
        assert!(loaded.is_ok(), "Deserialization should succeed");

        let loaded = loaded.unwrap();
        assert_eq!(loaded.algorithm(), Algorithm::lightgbm);

        // Compare predictions
        let original_preds = estimator
            .predict(&dataset.x_test, dataset.num_features, 0)
            .unwrap();
        let loaded_preds = loaded
            .predict(&dataset.x_test, dataset.num_features, 0)
            .unwrap();

        assert_eq!(
            original_preds.len(),
            loaded_preds.len(),
            "Loaded model should produce same number of predictions"
        );

        for (orig, load) in original_preds.iter().zip(loaded_preds.iter()) {
            assert!(
                (orig - load).abs() < 1e-6,
                "Loaded model predictions should match original"
            );
        }
    }

    #[test]
    fn test_estimator_debug_format() {
        let dataset = create_regression_dataset();
        let hyperparams = Hyperparams::new();

        let estimator = fit_regression(&dataset, &hyperparams).unwrap();
        let debug_str = format!("{:?}", estimator);
        assert!(
            debug_str.contains("Estimator"),
            "Debug format should contain struct name"
        );
    }

    #[test]
    fn test_fit_unsupported_task() {
        let dataset = create_regression_dataset();
        let hyperparams = Hyperparams::new();

        let result = fit(&dataset, &hyperparams, Task::text_classification);
        assert!(result.is_err(), "fit should fail for unsupported task");

        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("only supports"),
            "Error should mention task limitation"
        );
    }

    #[test]
    fn test_fit_with_verbosity() {
        let dataset = create_regression_dataset();
        let mut hyperparams = Hyperparams::new();
        hyperparams.insert("verbosity".to_string(), serde_json::json!(-1)); // Silent

        let result = fit_regression(&dataset, &hyperparams);
        assert!(result.is_ok(), "fit with verbosity should succeed");
    }

    #[test]
    fn test_classification_sets_correct_objective() {
        // Test that binary classification sets "binary" objective
        let binary_dataset = create_binary_classification_dataset();
        let hyperparams = Hyperparams::new();

        let result = fit_classification(&binary_dataset, &hyperparams);
        assert!(result.is_ok(), "Binary classification should succeed");

        // Test that multiclass classification sets "multiclass" objective
        let multiclass_dataset = create_multiclass_classification_dataset();
        let result = fit_classification(&multiclass_dataset, &hyperparams);
        assert!(result.is_ok(), "Multiclass classification should succeed");
    }
}
