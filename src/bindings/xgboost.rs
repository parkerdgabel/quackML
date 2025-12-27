use anyhow::Result;
use rand::*;
use xgboost::parameters::tree::*;
use xgboost::parameters::*;
use xgboost::{Booster, DMatrix};

use crate::orm::{Dataset, Hyperparams};

use super::Bindings;

fn get_dart_params(hyperparams: &Hyperparams) -> dart::DartBoosterParameters {
    let mut params = dart::DartBoosterParametersBuilder::default();
    for (key, value) in hyperparams {
        match key.as_str() {
            "rate_drop" => params.rate_drop(value.as_f64().unwrap() as f32),
            "one_drop" => params.one_drop(value.as_bool().unwrap()),
            "skip_drop" => params.skip_drop(value.as_f64().unwrap() as f32),
            "sample_type" => match value.as_str().unwrap() {
                "uniform" => params.sample_type(dart::SampleType::Uniform),
                "weighted" => params.sample_type(dart::SampleType::Weighted),
                _ => panic!("Unknown {:?}: {:?}", key, value),
            },
            "normalize_type" => match value.as_str().unwrap() {
                "tree" => params.normalize_type(dart::NormalizeType::Tree),
                "forest" => params.normalize_type(dart::NormalizeType::Forest),
                _ => panic!("Unknown {:?}: {:?}", key, value),
            },
            "booster" | "n_estimators" | "boost_rounds" => &mut params, // Valid but not relevant to this section
            "nthread" => &mut params,
            _ => panic!("Unknown {:?}: {:?}", key, value),
        };
    }
    params.build().unwrap()
}

fn get_linear_params(hyperparams: &Hyperparams) -> linear::LinearBoosterParameters {
    let mut params = linear::LinearBoosterParametersBuilder::default();
    for (key, value) in hyperparams {
        match key.as_str() {
            "alpha" | "reg_alpha" => params.alpha(value.as_f64().unwrap() as f32),
            "lambda" | "reg_lambda" => params.lambda(value.as_f64().unwrap() as f32),
            "updater" => match value.as_str().unwrap() {
                "shotgun" => params.updater(linear::LinearUpdate::Shotgun),
                "coord_descent" => params.updater(linear::LinearUpdate::CoordDescent),
                _ => panic!("Unknown {:?}: {:?}", key, value),
            },
            "booster" | "n_estimators" | "boost_rounds" => &mut params, // Valid but not relevant to this section
            "nthread" => &mut params,
            _ => panic!("Unknown {:?}: {:?}", key, value),
        };
    }
    params.build().unwrap()
}

fn get_tree_params(hyperparams: &Hyperparams) -> tree::TreeBoosterParameters {
    let mut params = tree::TreeBoosterParametersBuilder::default();
    for (key, value) in hyperparams {
        match key.as_str() {
            "eta" | "learning_rate" => params.eta(value.as_f64().unwrap() as f32),
            "gamma" | "min_split_loss" => params.gamma(value.as_f64().unwrap() as f32),
            "max_depth" => params.max_depth(value.as_u64().unwrap() as u32),
            "min_child_weight" => params.min_child_weight(value.as_f64().unwrap() as f32),
            "max_delta_step" => params.max_delta_step(value.as_f64().unwrap() as f32),
            "subsample" => params.subsample(value.as_f64().unwrap() as f32),
            "colsample_bytree" => params.colsample_bytree(value.as_f64().unwrap() as f32),
            "colsample_bylevel" => params.colsample_bylevel(value.as_f64().unwrap() as f32),
            "lambda" | "reg_lambda" => params.lambda(value.as_f64().unwrap() as f32),
            "alpha" | "reg_alpha" => params.alpha(value.as_f64().unwrap() as f32),
            "tree_method" => match value.as_str().unwrap() {
                "auto" => params.tree_method(TreeMethod::Auto),
                "exact" => params.tree_method(TreeMethod::Exact),
                "approx" => params.tree_method(TreeMethod::Approx),
                "hist" => params.tree_method(TreeMethod::Hist),
                "gpu_exact" => params.tree_method(TreeMethod::GpuExact),
                "gpu_hist" => params.tree_method(TreeMethod::GpuHist),
                _ => panic!("Unknown hyperparameter {:?}: {:?}", key, value),
            },
            "sketch_eps" => params.sketch_eps(value.as_f64().unwrap() as f32),
            "scale_pos_weight" => params.scale_pos_weight(value.as_f64().unwrap() as f32),
            "updater" => match value.as_array() {
                Some(array) => {
                    let mut v = Vec::new();
                    for value in array {
                        match value.as_str().unwrap() {
                            "grow_col_maker" => v.push(TreeUpdater::GrowColMaker),
                            "dist_col" => v.push(TreeUpdater::DistCol),
                            "grow_hist_maker" => v.push(TreeUpdater::GrowHistMaker),
                            "grow_local_hist_maker" => v.push(TreeUpdater::GrowLocalHistMaker),
                            "grow_sk_maker" => v.push(TreeUpdater::GrowSkMaker),
                            "sync" => v.push(TreeUpdater::Sync),
                            "refresh" => v.push(TreeUpdater::Refresh),
                            "prune" => v.push(TreeUpdater::Prune),
                            _ => panic!("Unknown hyperparameter {:?}: {:?}", key, value),
                        }
                    }
                    params.updater(v)
                }
                _ => panic!("updater should be a JSON array. Got: {:?}", value),
            },
            "refresh_leaf" => params.refresh_leaf(value.as_bool().unwrap()),
            "process_type" => match value.as_str().unwrap() {
                "default" => params.process_type(ProcessType::Default),
                "update" => params.process_type(ProcessType::Update),
                _ => panic!("Unknown hyperparameter {:?}: {:?}", key, value),
            },
            "grow_policy" => match value.as_str().unwrap() {
                "depthwise" => params.grow_policy(GrowPolicy::Depthwise),
                "loss_guide" => params.grow_policy(GrowPolicy::LossGuide),
                _ => panic!("Unknown hyperparameter {:?}: {:?}", key, value),
            },
            "predictor" => match value.as_str().unwrap() {
                "cpu" => params.predictor(Predictor::Cpu),
                "gpu" => params.predictor(Predictor::Gpu),
                _ => panic!("Unknown hyperparameter {:?}: {:?}", key, value),
            },
            "max_leaves" => params.max_leaves(value.as_u64().unwrap() as u32),
            "max_bin" => params.max_bin(value.as_u64().unwrap() as u32),
            "booster" | "n_estimators" | "boost_rounds" | "eval_metric" | "objective" => {
                &mut params
            } // Valid but not relevant to this section
            "nthread" => &mut params,
            "random_state" => &mut params,
            _ => panic!("Unknown hyperparameter {:?}: {:?}", key, value),
        };
    }
    params.build().unwrap()
}

pub fn fit_regression(dataset: &Dataset, hyperparams: &Hyperparams) -> Result<Box<dyn Bindings>> {
    fit(dataset, hyperparams, learning::Objective::RegLinear)
}

pub fn fit_classification(
    dataset: &Dataset,
    hyperparams: &Hyperparams,
) -> Result<Box<dyn Bindings>> {
    fit(
        dataset,
        hyperparams,
        learning::Objective::MultiSoftprob(dataset.num_distinct_labels.try_into().unwrap()),
    )
}

fn eval_metric_from_string(name: &str) -> learning::EvaluationMetric {
    match name {
        "rmse" => learning::EvaluationMetric::RMSE,
        "mae" => learning::EvaluationMetric::MAE,
        "logloss" => learning::EvaluationMetric::LogLoss,
        "merror" => learning::EvaluationMetric::MultiClassErrorRate,
        "mlogloss" => learning::EvaluationMetric::MultiClassLogLoss,
        "auc" => learning::EvaluationMetric::AUC,
        "ndcg" => learning::EvaluationMetric::NDCG,
        "ndcg-" => learning::EvaluationMetric::NDCGNegative,
        "map" => learning::EvaluationMetric::MAP,
        "map-" => learning::EvaluationMetric::MAPNegative,
        "poisson-nloglik" => learning::EvaluationMetric::PoissonLogLoss,
        "gamma-nloglik" => learning::EvaluationMetric::GammaLogLoss,
        "cox-nloglik" => learning::EvaluationMetric::CoxLogLoss,
        "gamma-deviance" => learning::EvaluationMetric::GammaDeviance,
        "tweedie-nloglik" => learning::EvaluationMetric::TweedieLogLoss,
        _ => panic!("Unknown eval_metric: {:?}", name),
    }
}

fn objective_from_string(name: &str, dataset: &Dataset) -> learning::Objective {
    match name {
        "reg:linear" => learning::Objective::RegLinear,
        "reg:logistic" => learning::Objective::RegLogistic,
        "binary:logistic" => learning::Objective::BinaryLogistic,
        "binary:logitraw" => learning::Objective::BinaryLogisticRaw,
        "gpu:reg:linear" => learning::Objective::GpuRegLinear,
        "gpu:reg:logistic" => learning::Objective::GpuRegLogistic,
        "gpu:binary:logistic" => learning::Objective::GpuBinaryLogistic,
        "gpu:binary:logitraw" => learning::Objective::GpuBinaryLogisticRaw,
        "count:poisson" => learning::Objective::CountPoisson,
        "survival:cox" => learning::Objective::SurvivalCox,
        "multi:softmax" => {
            learning::Objective::MultiSoftmax(dataset.num_distinct_labels.try_into().unwrap())
        }
        "multi:softprob" => {
            learning::Objective::MultiSoftprob(dataset.num_distinct_labels.try_into().unwrap())
        }
        "rank:pairwise" => learning::Objective::RankPairwise,
        "reg:gamma" => learning::Objective::RegGamma,
        "reg:tweedie" => learning::Objective::RegTweedie(Some(dataset.num_distinct_labels as f32)),
        _ => panic!("Unknown objective: {:?}", name),
    }
}

fn fit(
    dataset: &Dataset,
    hyperparams: &Hyperparams,
    objective: learning::Objective,
) -> Result<Box<dyn Bindings>> {
    // split the train/test data into DMatrix
    let mut dtrain = DMatrix::from_dense(&dataset.x_train, dataset.num_train_rows).unwrap();
    let mut dtest = DMatrix::from_dense(&dataset.x_test, dataset.num_test_rows).unwrap();
    dtrain.set_labels(&dataset.y_train).unwrap();
    dtest.set_labels(&dataset.y_test).unwrap();

    // specify datasets to evaluate against during training
    let evaluation_sets = &[(&dtrain, "train"), (&dtest, "test")];

    let seed = match hyperparams.get("random_state") {
        Some(value) => value.as_u64().unwrap(),
        None => 0,
    };
    let eval_metrics = match hyperparams.get("eval_metric") {
        Some(metrics) => {
            if metrics.is_array() {
                learning::Metrics::Custom(
                    metrics
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|metric| eval_metric_from_string(metric.as_str().unwrap()))
                        .collect(),
                )
            } else {
                learning::Metrics::Custom(Vec::from([eval_metric_from_string(
                    metrics.as_str().unwrap(),
                )]))
            }
        }
        None => learning::Metrics::Auto,
    };
    let learning_params = match learning::LearningTaskParametersBuilder::default()
        .objective(match hyperparams.get("objective") {
            Some(value) => objective_from_string(value.as_str().unwrap(), dataset),
            None => objective,
        })
        .eval_metrics(eval_metrics)
        .seed(seed)
        .build()
    {
        Ok(params) => params,
        Err(e) => panic!("Failed to parse learning params:\n\n{}", e),
    };

    // overall configuration for Booster
    let booster_params = match BoosterParametersBuilder::default()
        .learning_params(learning_params)
        .booster_type(match hyperparams.get("booster") {
            Some(value) => match value.as_str().unwrap() {
                "gbtree" => BoosterType::Tree(get_tree_params(hyperparams)),
                "linear" => BoosterType::Linear(get_linear_params(hyperparams)),
                "dart" => BoosterType::Dart(get_dart_params(hyperparams)),
                _ => panic!("Unknown booster: {:?}", value),
            },
            _ => BoosterType::Tree(get_tree_params(hyperparams)),
        })
        .threads(
            hyperparams
                .get("nthread")
                .map(|value| value.as_i64().expect("nthread must be an integer") as u32),
        )
        .verbose(true)
        .build()
    {
        Ok(params) => params,
        Err(e) => panic!("Failed to configure booster:\n\n{}", e),
    };

    let mut builder = TrainingParametersBuilder::default();
    // number of training iterations is aliased
    match hyperparams.get("n_estimators") {
        Some(value) => builder.boost_rounds(value.as_u64().unwrap() as u32),
        None => match hyperparams.get("boost_rounds") {
            Some(value) => builder.boost_rounds(value.as_u64().unwrap() as u32),
            None => &mut builder,
        },
    };

    let params = match builder
        // dataset to train with
        .dtrain(&dtrain)
        // optional datasets to evaluate against in each iteration
        .evaluation_sets(Some(evaluation_sets))
        // model parameters
        .booster_params(booster_params)
        .build()
    {
        Ok(params) => params,
        Err(e) => panic!("Failed to create training parameters:\n\n{}", e),
    };

    // train model, and print evaluation data
    let booster = match Booster::train(&params) {
        Ok(booster) => booster,
        Err(e) => panic!("Failed to train model:\n\n{}", e),
    };

    Ok(Box::new(Estimator { estimator: booster }))
}
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

#[cfg(test)]
mod tests {
    use super::*;
    use indexmap::IndexMap;

    /// Helper to create a simple regression dataset
    fn create_regression_dataset() -> Dataset {
        let x_train = vec![
            1.0, 2.0,
            2.0, 3.0,
            3.0, 4.0,
            4.0, 5.0,
            5.0, 6.0,
            6.0, 7.0,
            7.0, 8.0,
            8.0, 9.0,
        ];
        let y_train = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0];

        let x_test = vec![
            9.0, 10.0,
            10.0, 11.0,
        ];
        let y_test = vec![19.0, 21.0];

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
            num_distinct_labels: 0,
        }
    }

    /// Helper to create a binary classification dataset
    fn create_binary_classification_dataset() -> Dataset {
        let x_train = vec![
            1.0, 1.0,
            1.5, 1.2,
            0.8, 1.3,
            1.2, 0.9,
            5.0, 5.0,
            5.5, 5.2,
            4.8, 5.3,
            5.2, 4.9,
        ];
        let y_train = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let x_test = vec![
            1.1, 1.1,
            5.1, 5.1,
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
        let x_train = vec![
            1.0, 1.0,
            1.2, 0.9,
            0.9, 1.1,
            5.0, 1.0,
            5.2, 0.9,
            4.9, 1.1,
            3.0, 5.0,
            3.2, 4.9,
            2.9, 5.1,
        ];
        let y_train = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];

        let x_test = vec![
            1.1, 1.0,
            5.1, 1.0,
            3.0, 5.0,
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
    // Parameter Parsing Tests
    // ===================

    #[test]
    fn test_get_tree_params_default() {
        let hyperparams: Hyperparams = IndexMap::new();
        let params = get_tree_params(&hyperparams);
        // Should build without panicking
        assert!(format!("{:?}", params).contains("TreeBoosterParameters"));
    }

    #[test]
    fn test_get_tree_params_with_eta() {
        let mut hyperparams: Hyperparams = IndexMap::new();
        hyperparams.insert("eta".to_string(), serde_json::json!(0.1));
        let params = get_tree_params(&hyperparams);
        assert!(format!("{:?}", params).contains("TreeBoosterParameters"));
    }

    #[test]
    fn test_get_tree_params_with_learning_rate_alias() {
        let mut hyperparams: Hyperparams = IndexMap::new();
        hyperparams.insert("learning_rate".to_string(), serde_json::json!(0.05));
        let params = get_tree_params(&hyperparams);
        assert!(format!("{:?}", params).contains("TreeBoosterParameters"));
    }

    #[test]
    fn test_get_tree_params_with_max_depth() {
        let mut hyperparams: Hyperparams = IndexMap::new();
        hyperparams.insert("max_depth".to_string(), serde_json::json!(6));
        let params = get_tree_params(&hyperparams);
        assert!(format!("{:?}", params).contains("TreeBoosterParameters"));
    }

    #[test]
    fn test_get_linear_params_default() {
        let hyperparams: Hyperparams = IndexMap::new();
        let params = get_linear_params(&hyperparams);
        assert!(format!("{:?}", params).contains("LinearBoosterParameters"));
    }

    #[test]
    fn test_get_linear_params_with_alpha() {
        let mut hyperparams: Hyperparams = IndexMap::new();
        hyperparams.insert("alpha".to_string(), serde_json::json!(0.01));
        let params = get_linear_params(&hyperparams);
        assert!(format!("{:?}", params).contains("LinearBoosterParameters"));
    }

    #[test]
    fn test_get_dart_params_default() {
        let hyperparams: Hyperparams = IndexMap::new();
        let params = get_dart_params(&hyperparams);
        assert!(format!("{:?}", params).contains("DartBoosterParameters"));
    }

    #[test]
    fn test_get_dart_params_with_rate_drop() {
        let mut hyperparams: Hyperparams = IndexMap::new();
        hyperparams.insert("rate_drop".to_string(), serde_json::json!(0.1));
        let params = get_dart_params(&hyperparams);
        assert!(format!("{:?}", params).contains("DartBoosterParameters"));
    }

    // ===================
    // Eval Metric Tests
    // ===================

    #[test]
    fn test_eval_metric_from_string_rmse() {
        let metric = eval_metric_from_string("rmse");
        assert!(format!("{:?}", metric).contains("RMSE"));
    }

    #[test]
    fn test_eval_metric_from_string_logloss() {
        let metric = eval_metric_from_string("logloss");
        assert!(format!("{:?}", metric).contains("LogLoss"));
    }

    #[test]
    fn test_eval_metric_from_string_auc() {
        let metric = eval_metric_from_string("auc");
        assert!(format!("{:?}", metric).contains("AUC"));
    }

    #[test]
    #[should_panic(expected = "Unknown eval_metric")]
    fn test_eval_metric_from_string_invalid() {
        eval_metric_from_string("invalid_metric");
    }

    // ===================
    // Objective Tests
    // ===================

    #[test]
    fn test_objective_from_string_reg_linear() {
        let dataset = create_regression_dataset();
        let obj = objective_from_string("reg:linear", &dataset);
        assert!(format!("{:?}", obj).contains("RegLinear"));
    }

    #[test]
    fn test_objective_from_string_binary_logistic() {
        let dataset = create_binary_classification_dataset();
        let obj = objective_from_string("binary:logistic", &dataset);
        assert!(format!("{:?}", obj).contains("BinaryLogistic"));
    }

    #[test]
    fn test_objective_from_string_multi_softmax() {
        let dataset = create_multiclass_classification_dataset();
        let obj = objective_from_string("multi:softmax", &dataset);
        assert!(format!("{:?}", obj).contains("MultiSoftmax"));
    }

    #[test]
    #[should_panic(expected = "Unknown objective")]
    fn test_objective_from_string_invalid() {
        let dataset = create_regression_dataset();
        objective_from_string("invalid:objective", &dataset);
    }

    // ===================
    // Fit and Predict Tests
    // ===================

    #[test]
    fn test_fit_regression() {
        let dataset = create_regression_dataset();
        let mut hyperparams: Hyperparams = IndexMap::new();
        hyperparams.insert("n_estimators".to_string(), serde_json::json!(10));

        let model = fit_regression(&dataset, &hyperparams);
        assert!(model.is_ok(), "Failed to fit regression model: {:?}", model.err());

        let model = model.unwrap();
        let predictions = model.predict(&dataset.x_test, dataset.num_features, 0);
        assert!(predictions.is_ok());
        assert_eq!(predictions.unwrap().len(), dataset.num_test_rows);
    }

    #[test]
    fn test_fit_classification_binary() {
        let dataset = create_binary_classification_dataset();
        let mut hyperparams: Hyperparams = IndexMap::new();
        hyperparams.insert("n_estimators".to_string(), serde_json::json!(10));

        let model = fit_classification(&dataset, &hyperparams);
        assert!(model.is_ok(), "Failed to fit binary classification model: {:?}", model.err());

        let model = model.unwrap();
        let predictions = model.predict(&dataset.x_test, dataset.num_features, dataset.num_distinct_labels);
        assert!(predictions.is_ok());

        let preds = predictions.unwrap();
        assert_eq!(preds.len(), dataset.num_test_rows);

        // Predictions should be valid class indices
        for pred in preds {
            assert!(pred >= 0.0 && pred < dataset.num_distinct_labels as f32);
        }
    }

    #[test]
    fn test_fit_classification_multiclass() {
        let dataset = create_multiclass_classification_dataset();
        let mut hyperparams: Hyperparams = IndexMap::new();
        hyperparams.insert("n_estimators".to_string(), serde_json::json!(10));

        let model = fit_classification(&dataset, &hyperparams);
        assert!(model.is_ok(), "Failed to fit multiclass classification model: {:?}", model.err());

        let model = model.unwrap();
        let predictions = model.predict(&dataset.x_test, dataset.num_features, dataset.num_distinct_labels);
        assert!(predictions.is_ok());

        let preds = predictions.unwrap();
        assert_eq!(preds.len(), dataset.num_test_rows);

        // Predictions should be valid class indices
        for pred in preds {
            assert!(pred >= 0.0 && pred < dataset.num_distinct_labels as f32);
        }
    }

    #[test]
    fn test_predict_proba() {
        let dataset = create_binary_classification_dataset();
        let mut hyperparams: Hyperparams = IndexMap::new();
        hyperparams.insert("n_estimators".to_string(), serde_json::json!(10));

        let model = fit_classification(&dataset, &hyperparams).unwrap();
        let probas = model.predict_proba(&dataset.x_test, dataset.num_features);
        assert!(probas.is_ok());

        let probas = probas.unwrap();
        // For binary classification with softprob, we get 2 probabilities per sample
        assert_eq!(probas.len(), dataset.num_test_rows * dataset.num_distinct_labels);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let dataset = create_regression_dataset();
        let mut hyperparams: Hyperparams = IndexMap::new();
        hyperparams.insert("n_estimators".to_string(), serde_json::json!(10));

        let model = fit_regression(&dataset, &hyperparams).unwrap();

        // Get predictions before serialization
        let predictions_before = model.predict(&dataset.x_test, dataset.num_features, 0).unwrap();

        // Serialize and deserialize
        let bytes = model.to_bytes().expect("Failed to serialize");
        let restored = Estimator::from_bytes(&bytes).expect("Failed to deserialize");

        // Get predictions after deserialization
        let predictions_after = restored.predict(&dataset.x_test, dataset.num_features, 0).unwrap();

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
    fn test_estimator_debug() {
        let dataset = create_regression_dataset();
        let mut hyperparams: Hyperparams = IndexMap::new();
        hyperparams.insert("n_estimators".to_string(), serde_json::json!(5));

        let model = fit_regression(&dataset, &hyperparams).unwrap();
        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("Estimator"));
    }

    #[test]
    fn test_fit_with_custom_hyperparams() {
        let dataset = create_regression_dataset();
        let mut hyperparams: Hyperparams = IndexMap::new();
        hyperparams.insert("n_estimators".to_string(), serde_json::json!(5));
        hyperparams.insert("max_depth".to_string(), serde_json::json!(3));
        hyperparams.insert("eta".to_string(), serde_json::json!(0.3));
        hyperparams.insert("subsample".to_string(), serde_json::json!(0.8));

        let model = fit_regression(&dataset, &hyperparams);
        assert!(model.is_ok(), "Failed to fit with custom hyperparams: {:?}", model.err());
    }
}
