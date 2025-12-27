use std::convert::From;

use super::{Dataset, Hyperparams};
use anyhow::{bail, Result};
use linfa::prelude::Predict;
use linfa::traits::Fit;

use ndarray::{ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};

use super::Bindings;
use crate::orm::*;

#[derive(Debug, Serialize, Deserialize)]
pub struct LinearRegression {
    estimator: linfa_linear::FittedLinearRegression<f32>,
    num_features: usize,
}

impl LinearRegression {
    pub fn fit(dataset: &Dataset, hyperparams: &Hyperparams) -> Result<Box<dyn Bindings>>
    where
        Self: Sized,
    {
        let records = ArrayView2::from_shape(
            (dataset.num_train_rows, dataset.num_features),
            &dataset.x_train,
        )
        .unwrap();

        let targets = ArrayView1::from_shape(dataset.num_train_rows, &dataset.y_train).unwrap();

        let linfa_dataset = linfa::DatasetBase::from((records, targets));
        let mut estimator = linfa_linear::LinearRegression::default();

        for (key, value) in hyperparams {
            match key.as_str() {
                "fit_intercept" => {
                    estimator = estimator
                        .with_intercept(value.as_bool().expect("fit_intercept must be boolean"))
                }
                _ => bail!("Unknown {}: {:?}", key.as_str(), value),
            };
        }

        let estimator = estimator.fit(&linfa_dataset).unwrap();

        Ok(Box::new(LinearRegression {
            estimator,
            num_features: dataset.num_features,
        }))
    }
}

impl Bindings for LinearRegression {
    /// Predict a novel datapoint.
    fn predict(
        &self,
        features: &[f32],
        num_features: usize,
        _num_classes: usize,
    ) -> Result<Vec<f32>> {
        let records =
            ArrayView2::from_shape((features.len() / num_features, num_features), features)?;
        Ok(self.estimator.predict(records).targets.into_raw_vec())
    }

    /// Predict a novel datapoint.
    fn predict_proba(&self, _features: &[f32], _num_features: usize) -> Result<Vec<f32>> {
        bail!("predict_proba is currently only supported by the Python runtime.")
    }

    /// Deserialize self from bytes, with additional context
    fn from_bytes(bytes: &[u8]) -> Result<Box<dyn Bindings>>
    where
        Self: Sized,
    {
        let estimator: LinearRegression = rmp_serde::from_read(bytes)?;
        Ok(Box::new(estimator))
    }

    /// Serialize self to bytes
    fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(rmp_serde::to_vec(self)?)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LogisticRegression {
    estimator_binary: Option<linfa_logistic::FittedLogisticRegression<f32, i32>>,
    estimator_multi: Option<linfa_logistic::MultiFittedLogisticRegression<f32, i32>>,
    num_features: usize,
    num_distinct_labels: usize,
}

impl LogisticRegression {
    pub fn fit(dataset: &Dataset, hyperparams: &Hyperparams) -> Result<Box<dyn Bindings>>
    where
        Self: Sized,
    {
        let records = ArrayView2::from_shape(
            (dataset.num_train_rows, dataset.num_features),
            &dataset.x_train,
        )
        .unwrap();

        // Copy to convert to i32 because LogisticRegression doesn't continuous targets.
        let y_train: Vec<i32> = dataset.y_train.iter().map(|x| *x as i32).collect();
        let targets = ArrayView1::from_shape(dataset.num_train_rows, &y_train).unwrap();

        let linfa_dataset = linfa::DatasetBase::from((records, targets));

        if dataset.num_distinct_labels > 2 {
            let mut estimator = linfa_logistic::MultiLogisticRegression::default();

            for (key, value) in hyperparams {
                match key.as_str() {
                    "fit_intercept" => {
                        estimator = estimator
                            .with_intercept(value.as_bool().expect("fit_intercept must be boolean"))
                    }
                    "alpha" => {
                        estimator =
                            estimator.alpha(value.as_f64().expect("alpha must be a float") as f32)
                    }
                    "max_iterations" => {
                        estimator = estimator.max_iterations(
                            value.as_i64().expect("max_iterations must be an integer") as u64,
                        )
                    }
                    "gradient_tolerance" => {
                        estimator = estimator.gradient_tolerance(
                            value.as_f64().expect("gradient_tolerance must be a float") as f32,
                        )
                    }
                    _ => bail!("Unknown {}: {:?}", key.as_str(), value),
                };
            }

            let estimator = estimator.fit(&linfa_dataset).unwrap();

            Ok(Box::new(LogisticRegression {
                estimator_binary: None,
                estimator_multi: Some(estimator),
                num_features: dataset.num_features,
                num_distinct_labels: dataset.num_distinct_labels,
            }))
        } else {
            let mut estimator = linfa_logistic::LogisticRegression::default();

            for (key, value) in hyperparams {
                match key.as_str() {
                    "fit_intercept" => {
                        estimator = estimator
                            .with_intercept(value.as_bool().expect("fit_intercept must be boolean"))
                    }
                    "alpha" => {
                        estimator =
                            estimator.alpha(value.as_f64().expect("alpha must be a float") as f32)
                    }
                    "max_iterations" => {
                        estimator = estimator.max_iterations(
                            value.as_i64().expect("max_iterations must be an integer") as u64,
                        )
                    }
                    "gradient_tolerance" => {
                        estimator = estimator.gradient_tolerance(
                            value.as_f64().expect("gradient_tolerance must be a float") as f32,
                        )
                    }
                    _ => bail!("Unknown {}: {:?}", key.as_str(), value),
                };
            }

            let estimator = estimator.fit(&linfa_dataset).unwrap();

            Ok(Box::new(LogisticRegression {
                estimator_binary: Some(estimator),
                estimator_multi: None,
                num_features: dataset.num_features,
                num_distinct_labels: dataset.num_distinct_labels,
            }))
        }
    }
}

impl Bindings for LogisticRegression {
    fn predict_proba(&self, _features: &[f32], _num_features: usize) -> Result<Vec<f32>> {
        bail!("predict_proba is currently only supported by the Python runtime.")
    }

    fn predict(
        &self,
        features: &[f32],
        _num_features: usize,
        _num_classes: usize,
    ) -> Result<Vec<f32>> {
        let records = ArrayView2::from_shape(
            (features.len() / self.num_features, self.num_features),
            features,
        )?;

        Ok(if self.num_distinct_labels > 2 {
            self.estimator_multi
                .as_ref()
                .unwrap()
                .predict(records)
                .targets
                .into_raw_vec()
                .into_iter()
                .map(|x| x as f32)
                .collect()
        } else {
            self.estimator_binary
                .as_ref()
                .unwrap()
                .predict(records)
                .targets
                .into_raw_vec()
                .into_iter()
                .map(|x| x as f32)
                .collect()
        })
    }

    /// Deserialize self from bytes, with additional context
    fn from_bytes(bytes: &[u8]) -> Result<Box<dyn Bindings>>
    where
        Self: Sized,
    {
        let estimator: LogisticRegression = rmp_serde::from_read(bytes)?;
        Ok(Box::new(estimator))
    }

    /// Serialize self to bytes
    fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(rmp_serde::to_vec(self)?)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Svm {
    estimator: linfa_svm::Svm<f32, f32>,
    num_features: usize,
}

impl Svm {
    pub fn fit(dataset: &Dataset, hyperparams: &Hyperparams) -> Result<Box<dyn Bindings>> {
        let records = ArrayView2::from_shape(
            (dataset.num_train_rows, dataset.num_features),
            &dataset.x_train,
        )
        .unwrap();

        let targets = ArrayView1::from_shape(dataset.num_train_rows, &dataset.y_train).unwrap();

        let linfa_dataset = linfa::DatasetBase::from((records, targets));
        let mut estimator = linfa_svm::Svm::params();

        let mut hyperparams = hyperparams.clone();

        // Default to Gaussian kernel, all the others are deathly slow.
        if !hyperparams.contains_key(&String::from("kernel")) {
            hyperparams.insert("kernel".to_string(), serde_json::Value::from("rbf"));
        }

        for (key, value) in hyperparams {
            match key.as_str() {
                "eps" => {
                    estimator = estimator.eps(value.as_f64().expect("eps must be a float") as f32)
                }
                "shrinking" => {
                    estimator =
                        estimator.shrinking(value.as_bool().expect("shrinking must be a bool"))
                }
                "kernel" => {
                    match value.as_str().expect("kernel must be a string") {
                        "poli" => estimator = estimator.polynomial_kernel(3.0, 1.0), // degree = 3, c = 1.0 as per Scikit
                        "linear" => estimator = estimator.linear_kernel(),
                        "rbf" => estimator = estimator.gaussian_kernel(1e-7), // Default eps
                        value => bail!("Unknown kernel: {}", value),
                    }
                }
                _ => bail!("Unknown {}: {:?}", key, value),
            }
        }

        let estimator = estimator.fit(&linfa_dataset).unwrap();

        Ok(Box::new(Svm {
            estimator,
            num_features: dataset.num_features,
        }))
    }
}

impl Bindings for Svm {
    fn predict_proba(&self, _features: &[f32], _num_features: usize) -> Result<Vec<f32>> {
        bail!("predict_proba is currently only supported by the Python runtime.")
    }

    /// Predict a novel datapoint.
    fn predict(
        &self,
        features: &[f32],
        num_features: usize,
        _num_classes: usize,
    ) -> Result<Vec<f32>> {
        let records =
            ArrayView2::from_shape((features.len() / num_features, num_features), features)?;

        Ok(self.estimator.predict(records).targets.into_raw_vec())
    }

    /// Deserialize self from bytes, with additional context
    fn from_bytes(bytes: &[u8]) -> Result<Box<dyn Bindings>>
    where
        Self: Sized,
    {
        let estimator: Svm = rmp_serde::from_read(bytes)?;
        Ok(Box::new(estimator))
    }

    /// Serialize self to bytes
    fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(rmp_serde::to_vec(self)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;

    /// Helper to create a simple regression dataset
    /// y = 2*x1 + 3*x2 + 1
    fn create_regression_dataset() -> Dataset {
        let x_train = vec![
            1.0, 1.0,  // y = 2*1 + 3*1 + 1 = 6
            2.0, 1.0,  // y = 2*2 + 3*1 + 1 = 8
            1.0, 2.0,  // y = 2*1 + 3*2 + 1 = 9
            3.0, 2.0,  // y = 2*3 + 3*2 + 1 = 13
            2.0, 3.0,  // y = 2*2 + 3*3 + 1 = 14
        ];
        let y_train = vec![6.0, 8.0, 9.0, 13.0, 14.0];

        let x_test = vec![
            4.0, 1.0,  // y = 2*4 + 3*1 + 1 = 12
            1.0, 4.0,  // y = 2*1 + 3*4 + 1 = 15
        ];
        let y_test = vec![12.0, 15.0];

        Dataset {
            x_train,
            y_train,
            x_test,
            y_test,
            num_features: 2,
            num_labels: 1,
            num_rows: 7,
            num_train_rows: 5,
            num_test_rows: 2,
            num_distinct_labels: 0, // Not used for regression
        }
    }

    /// Helper to create a binary classification dataset
    fn create_binary_classification_dataset() -> Dataset {
        // Simple linearly separable data
        let x_train = vec![
            1.0, 1.0,   // class 0
            1.5, 1.2,   // class 0
            0.8, 1.3,   // class 0
            1.2, 0.9,   // class 0
            5.0, 5.0,   // class 1
            5.5, 5.2,   // class 1
            4.8, 5.3,   // class 1
            5.2, 4.9,   // class 1
        ];
        let y_train = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let x_test = vec![
            1.1, 1.1,  // should be class 0
            5.1, 5.1,  // should be class 1
        ];
        let y_test = vec![0.0, 1.0];

        Dataset {
            x_train,
            y_train,
            x_test,
            y_test,
            num_features: 2,
            num_labels: 1,
            num_rows: 10,
            num_train_rows: 8,
            num_test_rows: 2,
            num_distinct_labels: 2,
        }
    }

    /// Helper to create a multiclass classification dataset
    fn create_multiclass_classification_dataset() -> Dataset {
        // 3 classes, linearly separable
        let x_train = vec![
            1.0, 1.0,   // class 0
            1.2, 0.9,   // class 0
            0.9, 1.1,   // class 0
            5.0, 1.0,   // class 1
            5.2, 0.9,   // class 1
            4.9, 1.1,   // class 1
            3.0, 5.0,   // class 2
            3.2, 4.9,   // class 2
            2.9, 5.1,   // class 2
        ];
        let y_train = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let x_test = vec![
            1.1, 1.0,  // should be class 0
            5.1, 1.0,  // should be class 1
            3.0, 5.0,  // should be class 2
        ];
        let y_test = vec![0.0, 1.0, 2.0];

        Dataset {
            x_train,
            y_train,
            x_test,
            y_test,
            num_features: 2,
            num_labels: 1,
            num_rows: 12,
            num_train_rows: 9,
            num_test_rows: 3,
            num_distinct_labels: 3,
        }
    }

    // ===================
    // LinearRegression Tests
    // ===================

    #[test]
    fn test_linear_regression_fit_and_predict() {
        let dataset = create_regression_dataset();
        let hyperparams: Hyperparams = IndexMap::new();

        let model = LinearRegression::fit(&dataset, &hyperparams).expect("Failed to fit model");

        // Predict on test data
        let predictions = model
            .predict(&dataset.x_test, dataset.num_features, 0)
            .expect("Failed to predict");

        assert_eq!(predictions.len(), dataset.num_test_rows);

        // Check predictions are reasonable (within 20% of expected)
        for (i, pred) in predictions.iter().enumerate() {
            let expected = dataset.y_test[i];
            let error = (pred - expected).abs() / expected;
            assert!(
                error < 0.2,
                "Prediction {} = {}, expected {} (error: {:.2}%)",
                i, pred, expected, error * 100.0
            );
        }
    }

    #[test]
    fn test_linear_regression_with_hyperparams() {
        let dataset = create_regression_dataset();
        let mut hyperparams: Hyperparams = IndexMap::new();
        hyperparams.insert("fit_intercept".to_string(), serde_json::json!(true));

        let model = LinearRegression::fit(&dataset, &hyperparams).expect("Failed to fit model");
        let predictions = model
            .predict(&dataset.x_test, dataset.num_features, 0)
            .expect("Failed to predict");

        assert_eq!(predictions.len(), dataset.num_test_rows);
    }

    #[test]
    fn test_linear_regression_serialization() {
        let dataset = create_regression_dataset();
        let hyperparams: Hyperparams = IndexMap::new();

        let model = LinearRegression::fit(&dataset, &hyperparams).expect("Failed to fit model");

        // Get predictions before serialization
        let predictions_before = model
            .predict(&dataset.x_test, dataset.num_features, 0)
            .expect("Failed to predict");

        // Serialize and deserialize
        let bytes = model.to_bytes().expect("Failed to serialize");
        let restored = LinearRegression::from_bytes(&bytes).expect("Failed to deserialize");

        // Get predictions after deserialization
        let predictions_after = restored
            .predict(&dataset.x_test, dataset.num_features, 0)
            .expect("Failed to predict");

        // Predictions should be identical
        assert_eq!(predictions_before.len(), predictions_after.len());
        for (before, after) in predictions_before.iter().zip(predictions_after.iter()) {
            assert!(
                (before - after).abs() < 1e-6,
                "Serialization changed predictions: {} vs {}",
                before, after
            );
        }
    }

    #[test]
    fn test_linear_regression_predict_proba_unsupported() {
        let dataset = create_regression_dataset();
        let hyperparams: Hyperparams = IndexMap::new();

        let model = LinearRegression::fit(&dataset, &hyperparams).expect("Failed to fit model");

        let result = model.predict_proba(&dataset.x_test, dataset.num_features);
        assert!(result.is_err());
    }

    // ===================
    // LogisticRegression Tests
    // ===================

    #[test]
    fn test_logistic_regression_binary_classification() {
        let dataset = create_binary_classification_dataset();
        let hyperparams: Hyperparams = IndexMap::new();

        let model = LogisticRegression::fit(&dataset, &hyperparams).expect("Failed to fit model");

        let predictions = model
            .predict(&dataset.x_test, dataset.num_features, dataset.num_distinct_labels)
            .expect("Failed to predict");

        assert_eq!(predictions.len(), dataset.num_test_rows);

        // Check predictions match expected classes
        for (i, pred) in predictions.iter().enumerate() {
            let expected = dataset.y_test[i];
            assert_eq!(
                *pred as i32, expected as i32,
                "Prediction {} = {}, expected {}",
                i, pred, expected
            );
        }
    }

    #[test]
    fn test_logistic_regression_multiclass_classification() {
        let dataset = create_multiclass_classification_dataset();
        let hyperparams: Hyperparams = IndexMap::new();

        let model = LogisticRegression::fit(&dataset, &hyperparams).expect("Failed to fit model");

        let predictions = model
            .predict(&dataset.x_test, dataset.num_features, dataset.num_distinct_labels)
            .expect("Failed to predict");

        assert_eq!(predictions.len(), dataset.num_test_rows);

        // Predictions should be in valid class range
        for pred in predictions.iter() {
            assert!(*pred >= 0.0 && *pred < dataset.num_distinct_labels as f32);
        }
    }

    #[test]
    fn test_logistic_regression_with_hyperparams() {
        let dataset = create_binary_classification_dataset();
        let mut hyperparams: Hyperparams = IndexMap::new();
        hyperparams.insert("fit_intercept".to_string(), serde_json::json!(true));
        hyperparams.insert("alpha".to_string(), serde_json::json!(0.01));
        hyperparams.insert("max_iterations".to_string(), serde_json::json!(100));

        let model = LogisticRegression::fit(&dataset, &hyperparams).expect("Failed to fit model");
        let predictions = model
            .predict(&dataset.x_test, dataset.num_features, dataset.num_distinct_labels)
            .expect("Failed to predict");

        assert_eq!(predictions.len(), dataset.num_test_rows);
    }

    #[test]
    fn test_logistic_regression_serialization() {
        let dataset = create_binary_classification_dataset();
        let hyperparams: Hyperparams = IndexMap::new();

        let model = LogisticRegression::fit(&dataset, &hyperparams).expect("Failed to fit model");

        let predictions_before = model
            .predict(&dataset.x_test, dataset.num_features, dataset.num_distinct_labels)
            .expect("Failed to predict");

        let bytes = model.to_bytes().expect("Failed to serialize");
        let restored = LogisticRegression::from_bytes(&bytes).expect("Failed to deserialize");

        let predictions_after = restored
            .predict(&dataset.x_test, dataset.num_features, dataset.num_distinct_labels)
            .expect("Failed to predict");

        assert_eq!(predictions_before.len(), predictions_after.len());
        for (before, after) in predictions_before.iter().zip(predictions_after.iter()) {
            assert_eq!(*before as i32, *after as i32);
        }
    }

    // ===================
    // SVM Tests
    // ===================

    #[test]
    fn test_svm_fit_and_predict() {
        let dataset = create_regression_dataset();
        let hyperparams: Hyperparams = IndexMap::new();

        let model = Svm::fit(&dataset, &hyperparams).expect("Failed to fit model");

        let predictions = model
            .predict(&dataset.x_test, dataset.num_features, 0)
            .expect("Failed to predict");

        assert_eq!(predictions.len(), dataset.num_test_rows);
    }

    #[test]
    fn test_svm_with_kernel_hyperparams() {
        let dataset = create_regression_dataset();
        let mut hyperparams: Hyperparams = IndexMap::new();
        hyperparams.insert("kernel".to_string(), serde_json::json!("rbf"));

        let model = Svm::fit(&dataset, &hyperparams).expect("Failed to fit model");
        let predictions = model
            .predict(&dataset.x_test, dataset.num_features, 0)
            .expect("Failed to predict");

        assert_eq!(predictions.len(), dataset.num_test_rows);
    }

    #[test]
    fn test_svm_linear_kernel() {
        let dataset = create_regression_dataset();
        let mut hyperparams: Hyperparams = IndexMap::new();
        hyperparams.insert("kernel".to_string(), serde_json::json!("linear"));

        let model = Svm::fit(&dataset, &hyperparams).expect("Failed to fit model");
        let predictions = model
            .predict(&dataset.x_test, dataset.num_features, 0)
            .expect("Failed to predict");

        assert_eq!(predictions.len(), dataset.num_test_rows);
    }

    #[test]
    fn test_svm_serialization() {
        let dataset = create_regression_dataset();
        let hyperparams: Hyperparams = IndexMap::new();

        let model = Svm::fit(&dataset, &hyperparams).expect("Failed to fit model");

        let predictions_before = model
            .predict(&dataset.x_test, dataset.num_features, 0)
            .expect("Failed to predict");

        let bytes = model.to_bytes().expect("Failed to serialize");
        let restored = Svm::from_bytes(&bytes).expect("Failed to deserialize");

        let predictions_after = restored
            .predict(&dataset.x_test, dataset.num_features, 0)
            .expect("Failed to predict");

        assert_eq!(predictions_before.len(), predictions_after.len());
        for (before, after) in predictions_before.iter().zip(predictions_after.iter()) {
            assert!(
                (before - after).abs() < 1e-6,
                "Serialization changed predictions: {} vs {}",
                before, after
            );
        }
    }

    #[test]
    fn test_svm_predict_proba_unsupported() {
        let dataset = create_regression_dataset();
        let hyperparams: Hyperparams = IndexMap::new();

        let model = Svm::fit(&dataset, &hyperparams).expect("Failed to fit model");

        let result = model.predict_proba(&dataset.x_test, dataset.num_features);
        assert!(result.is_err());
    }

    #[test]
    fn test_svm_invalid_kernel() {
        let dataset = create_regression_dataset();
        let mut hyperparams: Hyperparams = IndexMap::new();
        hyperparams.insert("kernel".to_string(), serde_json::json!("invalid_kernel"));

        let result = Svm::fit(&dataset, &hyperparams);
        assert!(result.is_err());
    }
}
