use crate::bindings::Bindings;
use crate::orm::dataset::Dataset;
use crate::orm::task::Task;
use crate::orm::Hyperparams;

use anyhow::{anyhow, Result};
// use lightgbm;
// use serde_json::json;

pub struct Estimator {
    // estimator: lightgbm::Booster,
}

// unsafe impl Send for Estimator {}
// unsafe impl Sync for Estimator {}

impl std::fmt::Debug for Estimator {
    fn fmt(
        &self,
        formatter: &mut std::fmt::Formatter<'_>,
    ) -> std::result::Result<(), std::fmt::Error> {
        formatter.debug_struct("Estimator").finish()
    }
}

pub fn fit_regression(dataset: &Dataset, hyperparams: &Hyperparams) -> Result<Box<dyn Bindings>> {
    Err(anyhow!("unimplemented"))
}

pub fn fit_classification(
    dataset: &Dataset,
    hyperparams: &Hyperparams,
) -> Result<Box<dyn Bindings>> {
    Err(anyhow!("unimplemented"))
}

fn fit(dataset: &Dataset, hyperparams: &Hyperparams, task: Task) -> Result<Box<dyn Bindings>> {
    Err(anyhow!("unimplemented"))
}

impl Bindings for Estimator {
    /// Predict a set of datapoints.
    fn predict(
        &self,
        features: &[f32],
        num_features: usize,
        num_classes: usize,
    ) -> Result<Vec<f32>> {
        Err(anyhow!("unimplemented"))
    }

    // Predict the raw probability of classes for a classifier.
    fn predict_proba(&self, features: &[f32], num_features: usize) -> Result<Vec<f32>> {
        Err(anyhow!("unimplemented"))
    }

    /// Serialize self to bytes
    fn to_bytes(&self) -> Result<Vec<u8>> {
        Err(anyhow!("unimplemented"))
    }

    /// Deserialize self from bytes, with additional context
    fn from_bytes(bytes: &[u8]) -> Result<Box<dyn Bindings>>
    where
        Self: Sized,
    {
        Err(anyhow!("unimplemented"))
    }
}
