use std::convert::From;

use anyhow::{bail, Result};
use linfa::prelude::Predict;
use linfa::traits::Fit;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
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
        Err(anyhow::anyhow!("unimplemented"))
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
        Err(anyhow::anyhow!("unimplemented"))
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
        Err(anyhow::anyhow!("unimplemented"))
    }

    /// Serialize self to bytes
    fn to_bytes(&self) -> Result<Vec<u8>> {
        Err(anyhow::anyhow!("unimplemented"))
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
        Err(anyhow::anyhow!("unimplemented"))
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
        Err(anyhow::anyhow!("unimplemented"))
    }

    /// Deserialize self from bytes, with additional context
    fn from_bytes(bytes: &[u8]) -> Result<Box<dyn Bindings>>
    where
        Self: Sized,
    {
        Err(anyhow::anyhow!("unimplemented"))
    }

    /// Serialize self to bytes
    fn to_bytes(&self) -> Result<Vec<u8>> {
        Err(anyhow::anyhow!("unimplemented"))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Svm {
    estimator: linfa_svm::Svm<f32, f32>,
    num_features: usize,
}

impl Svm {
    pub fn fit(dataset: &Dataset, hyperparams: &Hyperparams) -> Result<Box<dyn Bindings>> {
        Err(anyhow::anyhow!("unimplemented"))
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
        Err(anyhow::anyhow!("unimplemented"))
    }

    /// Deserialize self from bytes, with additional context
    fn from_bytes(bytes: &[u8]) -> Result<Box<dyn Bindings>>
    where
        Self: Sized,
    {
        Err(anyhow::anyhow!("unimplemented"))
    }

    /// Serialize self to bytes
    fn to_bytes(&self) -> Result<Vec<u8>> {
        Err(anyhow::anyhow!("unimplemented"))
    }
}
