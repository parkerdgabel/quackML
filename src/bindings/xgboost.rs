use anyhow::Result;
use rand::*;
use xgboost::{Booster, DMatrix};

use super::Bindings;

pub struct Estimator {
    estimator: xgboost::Booster,
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

impl Bindings for Estimator {
    fn predict(
        &self,
        features: &[f32],
        num_features: usize,
        num_classes: usize,
    ) -> Result<Vec<f32>> {
        let x = DMatrix::from_dense(features, features.len() / num_features)?;
        let y = self.estimator.predict(&x)?;
        Ok(match num_classes {
            0 => y,
            _ => y
                .chunks(num_classes)
                .map(|probabilities| {
                    probabilities
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.total_cmp(b))
                        .map(|(index, _)| index)
                        .unwrap() as f32
                })
                .collect::<Vec<f32>>(),
        })
    }

    fn predict_proba(&self, features: &[f32], num_features: usize) -> Result<Vec<f32>> {
        let x = DMatrix::from_dense(features, features.len() / num_features)?;
        Ok(self.estimator.predict(&x)?)
    }

    /// Serialize self to bytes
    fn to_bytes(&self) -> Result<Vec<u8>> {
        let r: u64 = rand::random();
        let path = format!("/tmp/pgml_{}.bin", r);
        self.estimator.save(std::path::Path::new(&path))?;
        let bytes = std::fs::read(&path)?;
        std::fs::remove_file(&path)?;
        Ok(bytes)
    }

    /// Deserialize self from bytes, with additional context
    fn from_bytes(bytes: &[u8]) -> Result<Box<dyn Bindings>>
    where
        Self: Sized,
    {
        let mut estimator = Booster::load_buffer(bytes);
        if estimator.is_err() {
            // backward compatibility w/ 2.0.0
            estimator = Booster::load_buffer(&bytes[16..]);
        }

        let mut estimator = estimator?;

        estimator
            .set_param("nthread", &2.to_string())
            .expect("could not set nthread XGBoost parameter");

        Ok(Box::new(Estimator { estimator }))
    }
}
