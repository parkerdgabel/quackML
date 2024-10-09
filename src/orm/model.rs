use crate::bindings::Bindings;
use crate::context::DATABASE_CONTEXT;
use crate::orm::{snapshot, TextDatasetType};
use crate::{bindings::*, context};

use crate::orm::metrics::calculate_r2;

use ::linfa::prelude::*;
use chrono::{DateTime, Utc};
use duckdb::types::ValueRef;

use rand::seq::SliceRandom;

use anyhow::{anyhow, bail, Result};
use duckdb::{params, Rows};
use indexmap::IndexMap;
use itertools::{izip, Itertools};
use ndarray::ArrayView1;
use ndarray_stats::*;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt::{Display, Error, Formatter};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;

use super::{metrics, Algorithm, Dataset, Hyperparams, Project, Search, Snapshot, Status, Task};

#[allow(clippy::type_complexity)]
static DEPLOYED_MODELS_BY_ID: Lazy<Mutex<HashMap<i64, Arc<Model>>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[derive(Debug)]
pub struct Model {
    pub id: i64,
    pub project_id: i64,
    pub snapshot_id: i64,
    pub algorithm: Algorithm,
    pub hyperparams: Hyperparams,
    // pub runtime: Runtime,
    pub status: Status,
    pub metrics: Option<Value>,
    pub search: Option<Search>,
    pub search_params: Value,
    pub search_args: Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub project: Project,
    pub snapshot: Snapshot,
    pub bindings: Option<Box<dyn Bindings>>,
    pub num_classes: usize,
    pub num_features: usize,
}

impl Display for Model {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(
            f,
            "Model {{ id: {}, task: {:?}, algorithm: {:?} }}",
            self.id, self.project.task, self.algorithm
        )
    }
}

impl Model {
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        project: &Project,
        snapshot: &mut Snapshot,
        algorithm: Algorithm,
        hyperparams: &Hyperparams,
        search: Option<Search>,
        search_params: serde_json::Value,
        search_args: serde_json::Value,
    ) -> Result<Model> {
        let dataset = snapshot.tabular_dataset();
        let status = Status::in_progress;
        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
        let result = conn
        .query_row("
        INSERT INTO quackml.models (project_id, snapshot_id, algorithm, hyperparams, status, search, search_params, search_args, num_features)
          VALUES ($1, $2, $3, $4, cast($5 as status), $6, $7, $8, $9)
          RETURNING id, project_id, snapshot_id, algorithm, hyperparams, status, metrics, search, search_params, search_args, created_at, updated_at;",
          params![project.id, snapshot.id, algorithm.to_string(), serde_json::to_string(&hyperparams).unwrap(), status.to_string(), search.map(|s| s.to_string()), serde_json::to_string(&search_params).unwrap(), serde_json::to_string(&search_args).unwrap(), dataset.num_features as i64], |row| {
            let model = Model {
                id: row.get(0)?,
                project_id: row.get(1)?,
                snapshot_id: row.get(2)?,
                algorithm: Algorithm::from_str(&row.get::<_, String>(3)?).unwrap(),
                hyperparams: serde_json::from_str(&row.get::<_, String>(4).inspect(|j| println!("Hperparams: {:?}", j))?).inspect(|v| println!("After Hyperparams {:?}:" , v)).unwrap(),
                status: Status::from_str(&row.get::<_, String>(5)?).unwrap(),
                metrics: row.get::<_, Option<String>>(6)?.map(|s| serde_json::from_str(&s).unwrap()),
                search: row.get::<_, Option<String>>(7)?.inspect(|s| println!("{}", s)).map(|s| Search::from_str(&s)).transpose().unwrap(),
                search_params: serde_json::from_str(&row.get::<_, String>(8)?).unwrap(),
                search_args: serde_json::from_str(&row.get::<_, String>(9)?).unwrap(),
                created_at: row.get(10).unwrap(),
                updated_at: row.get(11).unwrap(),
                project: project.clone(),
                snapshot: snapshot.clone(),
                bindings: None,
                num_classes: match project.task {
                    Task::regression => 0,
                    _ => snapshot.num_classes(),
                },
                num_features: snapshot.num_features(),
            };
            Ok(model)
          });
        let mut model = result?;

        model.fit(&dataset);

        Ok(model)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn finetune(
        project: &Project,
        snapshot: &mut Snapshot,
        hyperparams: Value,
    ) -> Result<Model> {
        let dataset_args = hyperparams
            .get("dataset_args")
            .map(|v| v.clone())
            .unwrap_or_else(|| json!({}));

        // let dataset = snapshot.text_classification_dataset(dataset_args);
        let dataset = if project.task == Task::text_classification {
            TextDatasetType::TextClassification(snapshot.text_classification_dataset(&dataset_args))
        } else if project.task == Task::text_pair_classification {
            TextDatasetType::TextPairClassification(
                snapshot.text_pair_classification_dataset(&dataset_args),
            )
        } else if project.task == Task::conversation {
            TextDatasetType::Conversation(snapshot.conversation_dataset(&dataset_args))
        } else if project.task == Task::summarization {
            TextDatasetType::TextSummarization(snapshot.text_summarization_dataset(&dataset_args))
        } 
        else {
            return Err(anyhow!("Unsupported task for finetuning"));
        };
        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };

        let result = conn
        .query_row("
          INSERT INTO quackml.models (project_id, snapshot_id, algorithm, hyperparams, status, search, search_params, search_args, num_features)
          VALUES ($1, $2, $3, $4, cast($5 as status), $6, $7, $8, $9)
          RETURNING id, project_id, snapshot_id, algorithm, hyperparams, status, metrics, search, search_params, search_args, created_at, updated_at;",
          params![project.id, snapshot.id, Algorithm::transformers.to_string(), serde_json::to_string(&hyperparams).unwrap(), Status::in_progress.to_string(), None as Option<String>, "{}", "{}",dataset.num_features() as i64], |row| {
            let model = Model {
                id: row.get(0)?,
                project_id: row.get(1)?,
                snapshot_id: row.get(2)?,
                algorithm: Algorithm::from_str(&row.get::<_, String>(3)?).unwrap(),
                hyperparams: serde_json::from_str(&row.get::<_, String>(4)?).unwrap(),
                status: Status::from_str(&row.get::<_, String>(5)?).unwrap(),
                metrics: row.get::<_, Option<String>>(6)?.map(|s| serde_json::from_str(&s).unwrap()),
                search: row.get::<_, Option<String>>(7)?.map(|s| Search::from_str(s.as_str()).unwrap()),
                search_params: row.get::<_, Option<String>>(8).unwrap_or(Some("{}".to_string())).map(|s| serde_json::from_str(s.as_str())).unwrap().unwrap(),
                search_args: row.get::<_, Option<String>>(9).unwrap_or(Some("{}".to_string())).map(|s| serde_json::from_str(s.as_str())).unwrap().unwrap(),
                created_at: row.get(10).unwrap(),
                updated_at: row.get(11).unwrap(),
                project: project.clone(),
                snapshot: snapshot.clone(),
                bindings: None,
                num_classes: 0,
                num_features: snapshot.num_features(),
            };
            Ok(model)
          });
        let mut model = result?;

        let id = model.id;
        let path = std::path::PathBuf::from(format!("/tmp/quackml/models/{id}"));

        let metrics: HashMap<String, f64> = match dataset {
            TextDatasetType::TextClassification(dataset) => {
                transformers::finetune_text_classification(
                    &project.task,
                    dataset,
                    &model.hyperparams,
                    &path,
                    project.id,
                    model.id,
                )
                .expect("Failed to finetune text classification model")
            }
            TextDatasetType::TextPairClassification(dataset) => {
                transformers::finetune_text_pair_classification(
                    &project.task,
                    dataset,
                    &model.hyperparams,
                    &path,
                    project.id,
                    model.id,
                )
                .expect("Failed to finetune text pair classification model")
            }
            TextDatasetType::Conversation(dataset) => transformers::finetune_conversation(
                &project.task,
                dataset,
                &model.hyperparams,
                &path,
                project.id,
                model.id,
            )
            .expect("Failed to finetune conversation model"),
            TextDatasetType::TextSummarization(dataset) => {
                transformers::finetune_text_summarization(
                    &project.task,
                    dataset,
                    &model.hyperparams,
                    &path,
                    project.id,
                    model.id,
                )
                    .expect("Failed to finetune text summarization model")
            },
        };

        model.metrics = Some(json!(metrics));

        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };

        let result = conn.execute(
            "UPDATE quackml.models SET metrics = $1::JSON, status = $2 WHERE id = $3;",
            params![
                serde_json::to_string(&model.metrics).unwrap(),
                Status::successful.to_string(),
                model.id
            ],
        );

        if Ok(1) != result {
            bail!("Failed to update model metrics");
        }

        // Save the bindings.
        if path.is_dir() {
            for entry in std::fs::read_dir(&path)? {
                let path = entry?.path();

                if path.is_file() {
                    let bytes = std::fs::read(&path)?;

                    for (i, chunk) in bytes.chunks(100_000_000).enumerate() {
                        let result = conn
                        .execute(
                            "INSERT INTO quackml.files (model_id, path, part, data) VALUES($1, $2, $3, $4) RETURNING id;",
                            params![
                                model.id,
                                path.file_name().unwrap().to_str(),
                                i,
                                chunk
                            ],
                        );
                        result?;
                    }
                }
            }
        } else {
            return Err(anyhow!("Model checkpoint folder does not exist!"));
        }

        conn.execute(
            "UPDATE quackml.models SET status = $1 WHERE id = $2",
            params![Status::successful.to_string(), model.id],
        )?;

        Ok(model)
    }

    fn find(id: i64) -> Result<Model> {
        let result = context::run(|conn| {
            let res =conn
                .query_row(
                    "SELECT m.id, m.project_id, m.snapshot_id, m.algorithm, m.hyperparams, m.status, m.metrics, m.search, m.search_params, m.search_args, m.created_at, m.updated_at, f.data
                        FROM quackml.models m
                        JOIN quackml.files f ON m.id = f.model_id
                        WHERE m.id = $1;",
                    params![id],
                    |row| {
                        let model_id = row.get::<_,i64>(0)?;
                        let project_id = row.get::<_,i64>(1)?;
                        let project = Project::find(project_id).unwrap();
                        let snapshot_id = row.get::<_,i64>(2)?;
                        let snapshot = Snapshot::find(snapshot_id).unwrap();
                        let algorithm = Algorithm::from_str(&row.get::<_, String>(3)?).unwrap();
                        let data = row.get::<_, Vec<u8>>(12)?;
                        let num_features = snapshot.num_features();
                        let num_classes = match project.task {
                            Task::regression => 0,
                            _ => snapshot.num_classes(),
                        };
                        let bindings: Box<dyn Bindings> = match algorithm {
                            #[cfg(feature = "python")]
                            Algorithm::transformers => {
                                match project.task {
                                    Task::text_classification => {
                                        transformers::TextClassifier::from_id(model_id).unwrap()
                                    }
                                    Task::text_pair_classification => {
                                        // transformers::Estimator::from_bytes(&data).unwrap()
                                        todo!()
                                    }
                                    Task::conversation => {
                                        // transformers::Estimator::from_bytes(&data).unwrap()
                                        todo!()
                                    }
                                    _ => panic!("Unsupported task for transformers"),
                                }
                                                            
                            },
                            Algorithm::xgboost => xgboost::Estimator::from_bytes(&data).unwrap(),
                            Algorithm::lightgbm => lightgbm::Estimator::from_bytes(&data).unwrap(),
                            Algorithm::linear => match project.task {
                                Task::regression => linfa::LinearRegression::from_bytes(&data).unwrap(),
                                Task::classification => linfa::LogisticRegression::from_bytes(&data).unwrap(),
                                _ => panic!("Unsupported algorithm"),
                            },
                            Algorithm::svm => linfa::Svm::from_bytes(&data).unwrap(),
                            _ => panic!("Unsupported algorithm"),
                        };
                        let model = Model {
                            id: row.get(0)?,
                            project_id,
                            snapshot_id,
                            algorithm,
                            hyperparams: serde_json::from_str(&row.get::<_, String>(4)?).unwrap(),
                            status: Status::from_str(&row.get::<_, String>(5)?).unwrap(),
                            metrics: row.get::<_, Option<String>>(6)?.map(|s| serde_json::from_str(&s).unwrap()),
                            search: row.get::<_, duckdb::types::Value>(7).map(|v| match v {
                                duckdb::types::Value::Text(s) => Some(Search::from_str(s.as_str()).unwrap()),
                                _ => None,
                            }).unwrap(),
                            search_params: row.get::<_, String>(8).map(|s| serde_json::from_str(s.as_str())).unwrap().unwrap(),
                            search_args: row.get::<_, String>(9).map(|s| serde_json::from_str(s.as_str())).unwrap().unwrap(),
                            created_at: row.get(10).unwrap(),
                            updated_at: row.get(11).unwrap(),
                            project,
                            snapshot,
                            bindings: Some(bindings),
                            num_classes,
                            num_features,
                        };
                        Ok(model)
                    },
                )
                .map_err(|e| anyhow!("Failed to find model: {}", e))?;
            Ok(res)
        });
        result
    }

    pub fn find_cached(id: i64) -> Result<Arc<Model>> {
        {
            let models = DEPLOYED_MODELS_BY_ID.lock();
            if let Some(model) = models.get(&id) {
                return Ok(model.clone());
            }
        }

        let model = Arc::new(Model::find(id)?);
        let mut models = DEPLOYED_MODELS_BY_ID.lock();
        models.insert(id, Arc::clone(&model));
        Ok(model)
    }

    fn get_fit_function(&self) -> crate::bindings::Fit {
        match self.project.task {
            Task::regression => match self.algorithm {
                Algorithm::xgboost => xgboost::fit_regression,
                Algorithm::lightgbm => lightgbm::fit_regression,
                Algorithm::linear => linfa::LinearRegression::fit,
                Algorithm::svm => linfa::Svm::fit,
                Algorithm::lasso => sklearn::lasso_regression,
                Algorithm::elastic_net => sklearn::elastic_net_regression,
                Algorithm::ridge => sklearn::ridge_regression,
                Algorithm::random_forest => sklearn::random_forest_regression,
                Algorithm::orthogonal_matching_pursuit => {
                    sklearn::orthogonal_matching_pursuit_regression
                }
                Algorithm::bayesian_ridge => sklearn::bayesian_ridge_regression,
                Algorithm::automatic_relevance_determination => {
                    sklearn::automatic_relevance_determination_regression
                }
                Algorithm::stochastic_gradient_descent => {
                    sklearn::stochastic_gradient_descent_regression
                }
                Algorithm::passive_aggressive => sklearn::passive_aggressive_regression,
                Algorithm::ransac => sklearn::ransac_regression,
                Algorithm::theil_sen => sklearn::theil_sen_regression,
                Algorithm::huber => sklearn::huber_regression,
                Algorithm::quantile => sklearn::quantile_regression,
                Algorithm::kernel_ridge => sklearn::kernel_ridge_regression,
                Algorithm::gaussian_process => sklearn::gaussian_process_regression,
                Algorithm::nu_svm => sklearn::nu_svm_regression,
                Algorithm::ada_boost => sklearn::ada_boost_regression,
                Algorithm::bagging => sklearn::bagging_regression,
                Algorithm::extra_trees => sklearn::extra_trees_regression,
                Algorithm::gradient_boosting_trees => sklearn::gradient_boosting_trees_regression,
                Algorithm::hist_gradient_boosting => sklearn::hist_gradient_boosting_regression,
                Algorithm::least_angle => sklearn::least_angle_regression,
                Algorithm::lasso_least_angle => sklearn::lasso_least_angle_regression,
                Algorithm::linear_svm => sklearn::linear_svm_regression,
                Algorithm::catboost => sklearn::catboost_regression,
                _ => todo!("Unsupported regression algorithm: {:?}", self.algorithm),
            },
            Task::classification => match self.algorithm {
                Algorithm::xgboost => xgboost::fit_classification,
                Algorithm::lightgbm => lightgbm::fit_classification,
                Algorithm::linear => linfa::LogisticRegression::fit,
                Algorithm::svm => linfa::Svm::fit,
                Algorithm::ridge => sklearn::ridge_classification,
                Algorithm::random_forest => sklearn::random_forest_classification,
                Algorithm::stochastic_gradient_descent => {
                    sklearn::stochastic_gradient_descent_classification
                }
                Algorithm::perceptron => sklearn::perceptron_classification,
                Algorithm::passive_aggressive => sklearn::passive_aggressive_classification,
                Algorithm::gaussian_process => sklearn::gaussian_process,
                Algorithm::nu_svm => sklearn::nu_svm_classification,
                Algorithm::ada_boost => sklearn::ada_boost_classification,
                Algorithm::bagging => sklearn::bagging_classification,
                Algorithm::extra_trees => sklearn::extra_trees_classification,
                Algorithm::gradient_boosting_trees => {
                    sklearn::gradient_boosting_trees_classification
                }
                Algorithm::hist_gradient_boosting => sklearn::hist_gradient_boosting_classification,
                Algorithm::linear_svm => sklearn::linear_svm_classification,
                Algorithm::catboost => sklearn::catboost_classification,
                _ => todo!("Unsupported classification algorithm: {:?}", self.algorithm),
            },
            Task::clustering => match self.algorithm {
                Algorithm::affinity_propagation => sklearn::affinity_propagation,
                Algorithm::birch => sklearn::birch,
                Algorithm::kmeans => sklearn::kmeans,
                Algorithm::mini_batch_kmeans => sklearn::mini_batch_kmeans,
                Algorithm::mean_shift => sklearn::mean_shift,
                _ => todo!("Unsupported clustering algorithm: {:?}", self.algorithm),
            },
            Task::decomposition => match self.algorithm {
                Algorithm::pca => sklearn::pca,
                _ => todo!("Unsupported decomposition algorithm: {:?}", self.algorithm),
            },
            _ => todo!("Unsupported task: {:?}", self.project.task),
        }
    }

    /// Generates a complete list of hyperparams that should be tested
    /// by combining the self.search_params. When search params are empty,
    /// the set only contains the self.hyperparams.
    fn get_all_hyperparams(&self, n_iter: usize) -> Vec<Hyperparams> {
        // Gather all hyperparams
        let mut all_hyperparam_names = Vec::new();
        let mut all_hyperparam_values = Vec::new();
        for (key, value) in self.hyperparams.iter() {
            all_hyperparam_names.push(key.to_string());
            all_hyperparam_values.push(vec![value.clone()]);
        }
        for (key, values) in self.search_params.as_object().unwrap() {
            if all_hyperparam_names.contains(key) {
                panic!(
                    "`{key}` cannot be present in both hyperparams and search_params. Please choose one or the other."
                );
            }
            all_hyperparam_names.push(key.to_string());
            all_hyperparam_values.push(values.as_array().unwrap().to_vec());
        }

        // The search space is all possible combinations
        let all_hyperparam_values: Vec<Vec<serde_json::Value>> = all_hyperparam_values
            .into_iter()
            .multi_cartesian_product()
            .collect();
        let mut all_hyperparam_values = match self.search {
            Some(Search::random) => {
                // TODO support things like ranges to be random sampled
                let mut rng = &mut rand::thread_rng();
                all_hyperparam_values
                    .choose_multiple(&mut rng, n_iter)
                    .cloned()
                    .collect()
            }
            _ => all_hyperparam_values,
        };

        // Empty set for a run of only the default values
        if all_hyperparam_values.is_empty() {
            all_hyperparam_values.push(Vec::new());
        }

        // Construct sets of hyperparams from the values
        all_hyperparam_values
            .iter()
            .map(|hyperparam_values| {
                let mut hyperparams = Hyperparams::new();
                for (idx, value) in hyperparam_values.iter().enumerate() {
                    let name = all_hyperparam_names[idx].clone();
                    hyperparams.insert(name, value.clone());
                }
                hyperparams
            })
            .collect()
    }

    // The box is borrowed so that it may be reused by the caller
    #[allow(clippy::borrowed_box)]
    fn test(&self, dataset: &Dataset) -> IndexMap<String, f32> {
        // Test the estimator on the data
        let y_hat = self.predict_batch(&dataset.x_test).unwrap();
        let y_test = &dataset.y_test;

        // Calculate metrics to evaluate this estimator and its hyperparams
        let mut metrics = IndexMap::new();
        match self.project.task {
            Task::regression => {
                #[cfg(all(feature = "python", any(test, feature = "pg_test")))]
                {
                    let sklearn_metrics = sklearn::regression_metrics(y_test, &y_hat).unwrap();
                    metrics.insert("sklearn_r2".to_string(), sklearn_metrics["r2"]);
                    metrics.insert(
                        "sklearn_mean_absolute_error".to_string(),
                        sklearn_metrics["mae"],
                    );
                    metrics.insert(
                        "sklearn_mean_squared_error".to_string(),
                        sklearn_metrics["mse"],
                    );
                }

                let y_test = ArrayView1::from(&y_test);
                let y_hat = ArrayView1::from(&y_hat);

                metrics.insert("r2".to_string(), calculate_r2(&y_test, &y_hat));
                metrics.insert(
                    "mean_absolute_error".to_string(),
                    y_hat
                        .mean_absolute_error(&y_test)
                        .map(|v| v as f32)
                        .unwrap(),
                );
                metrics.insert(
                    "mean_squared_error".to_string(),
                    y_hat.mean_squared_error(&y_test).map(|v| v as f32).unwrap(),
                );
            }
            Task::classification => {
                #[cfg(all(feature = "python", any(test, feature = "pg_test")))]
                {
                    let sklearn_metrics = sklearn::classification_metrics(
                        y_test,
                        &y_hat,
                        dataset.num_distinct_labels,
                    )
                    .unwrap();

                    if dataset.num_distinct_labels == 2 {
                        metrics.insert("sklearn_roc_auc".to_string(), sklearn_metrics["roc_auc"]);
                    }

                    metrics.insert("sklearn_f1".to_string(), sklearn_metrics["f1"]);
                    metrics.insert("sklearn_f1_micro".to_string(), sklearn_metrics["f1_micro"]);
                    metrics.insert(
                        "sklearn_precision".to_string(),
                        sklearn_metrics["precision"],
                    );
                    metrics.insert("sklearn_recall".to_string(), sklearn_metrics["recall"]);
                    metrics.insert("sklearn_accuracy".to_string(), sklearn_metrics["accuracy"]);
                    metrics.insert("sklearn_mcc".to_string(), sklearn_metrics["mcc"]);

                    // You can always compare Scikit's confusion matrix to ours
                    // for debugging.
                    // let _sklearn_conf = crate::bindings::sklearn::confusion_matrix(&y_test, &y_hat);
                }

                if dataset.num_distinct_labels == 2 {
                    metrics.insert(
                        "roc_auc".to_string(),
                        metrics::roc_auc(
                            &ArrayView1::from(
                                &y_test.iter().map(|&v| v == 1.0).collect::<Vec<bool>>(),
                            ),
                            &ArrayView1::from(&y_hat),
                        ),
                    );
                    metrics.insert(
                        "log_loss".to_string(),
                        metrics::log_loss(
                            &ArrayView1::from(&y_test),
                            &ArrayView1::from(&y_hat),
                            1e-15,
                        ),
                    );
                }

                let y_hat: Vec<usize> = y_hat.iter().map(|&i| i.round() as usize).collect();
                let y_test: Vec<usize> = y_test.iter().map(|i| i.round() as usize).collect();
                let y_hat = ArrayView1::from(&y_hat);
                let y_test = ArrayView1::from(&y_test);

                // This one is buggy (Linfa).
                // let confusion_matrix = y_hat.confusion_matrix(y_test).unwrap();

                // This has to be identical to Scikit.
                let quackml_confusion_matrix = crate::orm::metrics::ConfusionMatrix::new(
                    &y_test,
                    &y_hat,
                    dataset.num_distinct_labels,
                );

                // These are validated against Scikit and seem to be correct.
                metrics.insert(
                    "f1".to_string(),
                    quackml_confusion_matrix.f1(crate::orm::metrics::Average::Macro),
                );
                metrics.insert(
                    "precision".to_string(),
                    quackml_confusion_matrix.precision(),
                );
                metrics.insert("recall".to_string(), quackml_confusion_matrix.recall());
                metrics.insert("accuracy".to_string(), quackml_confusion_matrix.accuracy());

                // This one is inaccurate, I have it in my TODO to reimplement.
                // metrics.insert("mcc".to_string(), confusion_matrix.mcc());
            }
            Task::clustering => {
                #[cfg(feature = "python")]
                {
                    let sklearn_metrics =
                        sklearn::clustering_metrics(dataset.num_features, &dataset.x_test, &y_hat)
                            .unwrap();
                    metrics.insert("silhouette".to_string(), sklearn_metrics["silhouette"]);
                }
            }
            Task::decomposition => {
                #[cfg(feature = "python")]
                {
                    let sklearn_metrics =
                        sklearn::decomposition_metrics(self.bindings.as_ref().unwrap()).unwrap();
                    metrics.insert(
                        "cumulative_explained_variance".to_string(),
                        sklearn_metrics["cumulative_explained_variance"],
                    );
                }
            }
            task => panic!("No test metrics available for task: {:?}", task),
        }

        metrics
    }

    fn get_bindings_and_metrics(
        &mut self,
        dataset: &Dataset,
        hyperparams: &Hyperparams,
    ) -> (Box<dyn Bindings>, IndexMap<String, f32>) {
        let fit = self.get_fit_function();
        let now = Instant::now();
        self.bindings = Some(fit(dataset, hyperparams).unwrap());
        let fit_time = now.elapsed();

        let now = Instant::now();
        let mut metrics = self.test(dataset);
        let score_time = now.elapsed();

        metrics.insert("fit_time".to_string(), fit_time.as_secs_f32());
        metrics.insert("score_time".to_string(), score_time.as_secs_f32());

        let mut bindings = None;
        std::mem::swap(&mut self.bindings, &mut bindings);
        (bindings.unwrap(), metrics)
    }

    pub fn fit_time(&self) -> f32 {
        self.metrics
            .as_ref()
            .unwrap()
            .get("fit_time")
            .unwrap()
            .as_f64()
            .unwrap() as f32
    }

    pub fn score_time(&self) -> f32 {
        self.metrics
            .as_ref()
            .unwrap()
            .get("score_time")
            .unwrap()
            .as_f64()
            .unwrap() as f32
    }

    pub fn f1(&self) -> f32 {
        self.metrics
            .as_ref()
            .unwrap()
            .get("f1")
            .unwrap()
            .as_f64()
            .unwrap() as f32
    }

    pub fn r2(&self) -> f32 {
        self.metrics
            .as_ref()
            .unwrap()
            .get("r2")
            .unwrap()
            .as_f64()
            .unwrap() as f32
    }

    fn fit(&mut self, dataset: &Dataset) {
        // Sometimes our algorithms take a long time. The only way to stop code
        // that we don't have control over is using a signal handler. Signal handlers
        // however are not allowed to allocate any memory. Therefore, we cannot register
        // a SIGINT query cancellation signal and return the connection to a healthy state
        // safely. The only way to cancel a training job then is to send a SIGTERM with
        // `SELECT pg_terminate_backend(pid)` which will process the interrupt, clean up,
        // and close the connection without affecting the postmaster.
        let signal_id = unsafe {
            signal_hook::low_level::register(signal_hook::consts::SIGTERM, || {
                // There can be no memory allocations here.
                // check_for_interrupts!();
            })
        }
        .unwrap();

        let mut n_iter: usize = 10;
        let mut cv: usize = if self.search.is_some() { 5 } else { 1 };
        for (key, value) in self.search_args.as_object().unwrap() {
            match key.as_str() {
                "n_iter" => n_iter = value.as_i64().unwrap().try_into().unwrap(),
                "cv" => cv = value.as_i64().unwrap().try_into().unwrap(),
                _ => panic!("Unknown search_args => {:?}: {:?}", key, value),
            }
        }

        let mut all_hyperparams = self.get_all_hyperparams(n_iter);
        let mut all_bindings = Vec::with_capacity(all_hyperparams.len());
        let mut all_metrics = Vec::with_capacity(all_hyperparams.len());

        // Train and score all the hyperparams on the dataset
        if cv < 2 {
            for hyperparams in &all_hyperparams {
                let (bindings, metrics) = self.get_bindings_and_metrics(dataset, hyperparams);
                all_bindings.push(bindings);
                all_metrics.push(metrics);
            }
        } else {
            // With 2 or more folds, generated for cross validation
            for k in 0..cv {
                let fold = dataset.fold(k, cv);
                for hyperparams in &all_hyperparams {
                    let (bindings, metrics) = self.get_bindings_and_metrics(&fold, hyperparams);
                    all_bindings.push(bindings);
                    all_metrics.push(metrics);
                }
            }
        }

        // Phew, we're done.
        signal_hook::low_level::unregister(signal_id);

        if all_metrics.len() == 1 {
            self.bindings = Some(all_bindings.pop().unwrap());
            self.hyperparams = all_hyperparams.pop().unwrap();
            self.metrics = Some(json!(all_metrics.pop().unwrap()));
        } else {
            let mut search_results = IndexMap::new();
            search_results.insert("params".to_string(), json!(all_hyperparams));
            search_results.insert("n_splits".to_string(), json!(cv));

            // Find the best estimator, hyperparams and metrics
            let target_metric = self.project.task.default_target_metric();
            let mut i = 0;
            let mut best_index = 0;
            let mut best_metric = f32::NEG_INFINITY;
            let mut best_metrics = None;
            let mut best_hyperparams = None;
            let mut best_estimator = None;
            let mut fit_times: Vec<Vec<f32>> = vec![vec![0.; cv]; all_hyperparams.len()];
            let mut score_times: Vec<Vec<f32>> = vec![vec![0.; cv]; all_hyperparams.len()];
            let mut test_scores: Vec<Vec<f32>> = vec![vec![0.; cv]; all_hyperparams.len()];
            let mut fold_scores: Vec<Vec<f32>> = vec![vec![0.; all_hyperparams.len()]; cv];
            #[allow(clippy::explicit_counter_loop)]
            for (metrics, estimator) in izip!(all_metrics, all_bindings) {
                let fold_i = i / all_hyperparams.len();
                let hyperparams_i = i % all_hyperparams.len();
                let hyperparams = &all_hyperparams[hyperparams_i];
                let metric = *metrics.get(&target_metric).unwrap();
                fit_times[hyperparams_i][fold_i] = *metrics.get("fit_time").unwrap();
                score_times[hyperparams_i][fold_i] = *metrics.get("score_time").unwrap();
                test_scores[hyperparams_i][fold_i] = metric;
                fold_scores[fold_i][hyperparams_i] = metric;

                if metric > best_metric {
                    best_index = hyperparams_i;
                    best_metric = metric;
                    best_metrics = Some(metrics);
                    best_hyperparams = Some(hyperparams);
                    best_estimator = Some(estimator);
                }
                i += 1;
            }

            search_results.insert("best_index".to_string(), json!(best_index));
            search_results.insert(
                "mean_fit_time".to_string(),
                json!(fit_times
                    .iter()
                    .map(|v| ArrayView1::from(v).mean().unwrap())
                    .collect::<Vec<f32>>()),
            );
            search_results.insert(
                "std_fit_time".to_string(),
                json!(fit_times
                    .iter()
                    .map(|v| ArrayView1::from(v).std(0.))
                    .collect::<Vec<f32>>()),
            );
            search_results.insert(
                "mean_score_time".to_string(),
                json!(score_times
                    .iter()
                    .map(|v| ArrayView1::from(v).mean().unwrap())
                    .collect::<Vec<f32>>()),
            );
            search_results.insert(
                "std_score_time".to_string(),
                json!(score_times
                    .iter()
                    .map(|v| ArrayView1::from(v).std(0.))
                    .collect::<Vec<f32>>()),
            );
            search_results.insert(
                "mean_test_score".to_string(),
                json!(test_scores
                    .iter()
                    .map(|v| ArrayView1::from(v).mean().unwrap())
                    .collect::<Vec<f32>>()),
            );
            search_results.insert(
                "std_test_score".to_string(),
                json!(test_scores
                    .iter()
                    .map(|v| ArrayView1::from(v).std(0.))
                    .collect::<Vec<f32>>()),
            );
            for (k, score) in fold_scores.iter().enumerate() {
                search_results.insert(format!("split{k}_test_score"), json!(score));
            }
            for param in best_hyperparams.unwrap().keys() {
                let params: Vec<serde_json::Value> = all_hyperparams
                    .iter()
                    .map(|hyperparams| json!(hyperparams.get(param).unwrap()))
                    .collect();
                search_results.insert(format!("param_{param}"), json!(params));
            }
            let mut metrics = IndexMap::new();
            for (key, value) in best_metrics.as_ref().unwrap() {
                metrics.insert(key.to_string(), json!(value));
            }
            metrics.insert("search_results".to_string(), json!(search_results));

            self.bindings = best_estimator;
            self.hyperparams = best_hyperparams.unwrap().clone();
            self.metrics = Some(json!(metrics));
        };
        println!("Hyperparams: {:?}", self.hyperparams);
        println!("Metrics: {:?}", self.metrics);
        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };

        conn.execute(
            "UPDATE quackml.models SET hyperparams = $1, metrics = $2 WHERE id = $3",
            params![
                serde_json::to_string(&self.hyperparams).unwrap(),
                serde_json::to_string(self.metrics.as_ref().unwrap()).unwrap(),
                self.id
            ],
        )
        .unwrap();

        // Save the bindings.
        conn.execute(
            "INSERT INTO quackml.files (model_id, path, part, data) VALUES($1, 'estimator.rmp', 0, $2)",
            params![
                self.id,
                self.bindings.as_ref().unwrap().to_bytes().unwrap()
            ],
        ).unwrap();
    }

    pub fn numeric_encode_features(&self, rows: &mut Rows) -> Vec<f32> {
        let mut features = Vec::new();
        while let Some(row) = rows.next().unwrap() {
            for (i, column) in self.snapshot.features().enumerate() {
                let attribute = row.get_ref(i).unwrap();
                match &column.statistics.categories {
                    Some(_categories) => {
                        let key = match attribute {
                            ValueRef::Null => snapshot::NULL_CATEGORY_KEY.to_string(),
                            ValueRef::Text(s) => String::from_utf8_lossy(s).to_string(),
                            ValueRef::Boolean(b) => b.to_string(),
                            ValueRef::SmallInt(i) => i.to_string(),
                            ValueRef::Int(i) => i.to_string(),
                            ValueRef::BigInt(i) => i.to_string(),
                            ValueRef::Float(f) => f.to_string(),
                            ValueRef::Double(d) => d.to_string(),
                            ValueRef::Decimal(d) => d.to_string(),
                            _ => snapshot::NULL_CATEGORY_KEY.to_string(),
                        };
                        let value = column.get_category_value(&key);
                        features.push(value);
                    }
                    None => {
                        match attribute {
                            ValueRef::Null => {
                                features.push(f32::NAN);
                            }
                            ValueRef::Boolean(b) => features.push(b as u8 as f32),
                            ValueRef::SmallInt(i) => features.push(i as f32),
                            ValueRef::Int(i) => features.push(i as f32),
                            ValueRef::BigInt(i) => features.push(i as f32),
                            ValueRef::Float(f) => features.push(f),
                            ValueRef::Double(d) => features.push(d as f32),
                            ValueRef::Decimal(d) => features.push(d.try_into().unwrap()),
                            ValueRef::Blob(b) => {
                                // Assuming Blob represents a list of numbers
                                let list: Vec<f32> = b
                                    .chunks(4)
                                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                                    .collect();
                                features.extend(list);
                            }
                            _ => eprintln!(
                                "Unsupported type for quantitative column: {:?}. type: {:?}",
                                column.name, attribute
                            ),
                        }
                    }
                }
            }
        }
        features
    }

    pub fn predict(&self, features: &[f32]) -> Result<f32> {
        Ok(self.predict_batch(features)?[0])
    }

    pub fn predict_proba(&self, features: &[f32]) -> Result<Vec<f32>> {
        match self.project.task {
            Task::regression => bail!("You can't predict probabilities for a regression model"),
            Task::classification => self
                .bindings
                .as_ref()
                .unwrap()
                .predict_proba(features, self.num_features),
            _ => bail!("no predict_proba for huggingface"),
        }
    }

    pub fn predict_joint(&self, features: &[f32]) -> Result<Vec<f32>> {
        match self.project.task {
            Task::regression => self.bindings.as_ref().unwrap().predict(
                features,
                self.num_features,
                self.num_classes,
            ),
            Task::classification => {
                bail!("You can't predict joint probabilities for a classification model")
            }
            _ => bail!("no predict_joint for huggingface"),
        }
    }

    pub fn predict_batch(&self, features: &[f32]) -> Result<Vec<f32>> {
        self.bindings
            .as_ref()
            .unwrap()
            .predict(features, self.num_features, self.num_classes)
    }

    pub fn decompose(&self, vector: &[f32]) -> Result<Vec<f32>> {
        self.bindings
            .as_ref()
            .unwrap()
            .predict(vector, self.num_features, self.num_classes)
    }
}
