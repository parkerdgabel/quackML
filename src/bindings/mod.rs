use std::any::Any;

use anyhow::Result;
use std::fmt::Debug;

pub mod lightgbm;
pub mod linfa;
pub mod transformers;
pub mod xgboost;

pub trait AToAny: 'static {
    fn as_any(&self) -> &dyn Any;
}

impl<T: 'static> AToAny for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}
/// The Bindings trait that has to be implemented by all algorithm
/// providers we use in PostgresML. We don't rely on Serde serialization,
/// since scikit-learn estimators were originally serialized in pure Python as
/// pickled objects, and neither xgboost nor linfa estimators completely
/// implement serde.
pub trait Bindings: Send + Sync + Debug + AToAny {
    /// Predict a set of datapoints.
    fn predict(
        &self,
        features: &[f32],
        num_features: usize,
        num_classes: usize,
    ) -> Result<Vec<f32>>;

    /// Predict the probability of each class.
    fn predict_proba(&self, features: &[f32], num_features: usize) -> Result<Vec<f32>>;

    /// Serialize self to bytes
    fn to_bytes(&self) -> Result<Vec<u8>>;

    /// Deserialize self from bytes, with additional context
    fn from_bytes(bytes: &[u8]) -> Result<Box<dyn Bindings>>
    where
        Self: Sized;
}
