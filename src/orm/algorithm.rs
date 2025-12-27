#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[allow(non_camel_case_types)]
pub enum Algorithm {
    linear,
    xgboost,
    xgboost_random_forest,
    svm,
    lasso,
    elastic_net,
    ridge,
    kmeans,
    dbscan,
    knn,
    random_forest,
    least_angle,
    lasso_least_angle,
    orthogonal_matching_pursuit,
    bayesian_ridge,
    automatic_relevance_determination,
    stochastic_gradient_descent,
    perceptron,
    passive_aggressive,
    ransac,
    theil_sen,
    huber,
    quantile,
    kernel_ridge,
    gaussian_process,
    nu_svm,
    ada_boost,
    bagging,
    extra_trees,
    gradient_boosting_trees,
    hist_gradient_boosting,
    linear_svm,
    lightgbm,
    transformers,
    affinity_propagation,
    birch,
    feature_agglomeration,
    mini_batch_kmeans,
    mean_shift,
    optics,
    spectral,
    spectral_bi,
    spectral_co,
    catboost,
    pca,
}

impl std::str::FromStr for Algorithm {
    type Err = ();

    fn from_str(input: &str) -> Result<Algorithm, Self::Err> {
        match input {
            "linear" => Ok(Algorithm::linear),
            "xgboost" => Ok(Algorithm::xgboost),
            "xgboost_random_forest" => Ok(Algorithm::xgboost_random_forest),
            "svm" => Ok(Algorithm::svm),
            "lasso" => Ok(Algorithm::lasso),
            "elastic_net" => Ok(Algorithm::elastic_net),
            "ridge" => Ok(Algorithm::ridge),
            "kmeans" => Ok(Algorithm::kmeans),
            "dbscan" => Ok(Algorithm::dbscan),
            "knn" => Ok(Algorithm::knn),
            "random_forest" => Ok(Algorithm::random_forest),
            "least_angle" => Ok(Algorithm::least_angle),
            "lasso_least_angle" => Ok(Algorithm::lasso_least_angle),
            "orthogonal_matching_pursuit" => Ok(Algorithm::orthogonal_matching_pursuit),
            "bayesian_ridge" => Ok(Algorithm::bayesian_ridge),
            "automatic_relevance_determination" => Ok(Algorithm::automatic_relevance_determination),
            "stochastic_gradient_descent" => Ok(Algorithm::stochastic_gradient_descent),
            "perceptron" => Ok(Algorithm::perceptron),
            "passive_aggressive" => Ok(Algorithm::passive_aggressive),
            "ransac" => Ok(Algorithm::ransac),
            "theil_sen" => Ok(Algorithm::theil_sen),
            "huber" => Ok(Algorithm::huber),
            "quantile" => Ok(Algorithm::quantile),
            "kernel_ridge" => Ok(Algorithm::kernel_ridge),
            "gaussian_process" => Ok(Algorithm::gaussian_process),
            "nu_svm" => Ok(Algorithm::nu_svm),
            "ada_boost" => Ok(Algorithm::ada_boost),
            "bagging" => Ok(Algorithm::bagging),
            "extra_trees" => Ok(Algorithm::extra_trees),
            "gradient_boosting_trees" => Ok(Algorithm::gradient_boosting_trees),
            "hist_gradient_boosting" => Ok(Algorithm::hist_gradient_boosting),
            "linear_svm" => Ok(Algorithm::linear_svm),
            "lightgbm" => Ok(Algorithm::lightgbm),
            "transformers" => Ok(Algorithm::transformers),
            "affinity_propagation" => Ok(Algorithm::affinity_propagation),
            "birch" => Ok(Algorithm::birch),
            "feature_agglomeration" => Ok(Algorithm::feature_agglomeration),
            "mini_batch_kmeans" => Ok(Algorithm::mini_batch_kmeans),
            "mean_shift" => Ok(Algorithm::mean_shift),
            "optics" => Ok(Algorithm::optics),
            "spectral" => Ok(Algorithm::spectral),
            "spectral_bi" => Ok(Algorithm::spectral_bi),
            "spectral_co" => Ok(Algorithm::spectral_co),
            "catboost" => Ok(Algorithm::catboost),
            "pca" => Ok(Algorithm::pca),
            _ => Err(()),
        }
    }
}

impl std::string::ToString for Algorithm {
    fn to_string(&self) -> String {
        match *self {
            Algorithm::linear => "linear".to_string(),
            Algorithm::xgboost => "xgboost".to_string(),
            Algorithm::xgboost_random_forest => "xgboost_random_forest".to_string(),
            Algorithm::svm => "svm".to_string(),
            Algorithm::lasso => "lasso".to_string(),
            Algorithm::elastic_net => "elastic_net".to_string(),
            Algorithm::ridge => "ridge".to_string(),
            Algorithm::kmeans => "kmeans".to_string(),
            Algorithm::dbscan => "dbscan".to_string(),
            Algorithm::knn => "knn".to_string(),
            Algorithm::random_forest => "random_forest".to_string(),
            Algorithm::least_angle => "least_angle".to_string(),
            Algorithm::lasso_least_angle => "lasso_least_angle".to_string(),
            Algorithm::orthogonal_matching_pursuit => "orthogonal_matching_pursuit".to_string(),
            Algorithm::bayesian_ridge => "bayesian_ridge".to_string(),
            Algorithm::automatic_relevance_determination => {
                "automatic_relevance_determination".to_string()
            }
            Algorithm::stochastic_gradient_descent => "stochastic_gradient_descent".to_string(),
            Algorithm::perceptron => "perceptron".to_string(),
            Algorithm::passive_aggressive => "passive_aggressive".to_string(),
            Algorithm::ransac => "ransac".to_string(),
            Algorithm::theil_sen => "theil_sen".to_string(),
            Algorithm::huber => "huber".to_string(),
            Algorithm::quantile => "quantile".to_string(),
            Algorithm::kernel_ridge => "kernel_ridge".to_string(),
            Algorithm::gaussian_process => "gaussian_process".to_string(),
            Algorithm::nu_svm => "nu_svm".to_string(),
            Algorithm::ada_boost => "ada_boost".to_string(),
            Algorithm::bagging => "bagging".to_string(),
            Algorithm::extra_trees => "extra_trees".to_string(),
            Algorithm::gradient_boosting_trees => "gradient_boosting_trees".to_string(),
            Algorithm::hist_gradient_boosting => "hist_gradient_boosting".to_string(),
            Algorithm::linear_svm => "linear_svm".to_string(),
            Algorithm::lightgbm => "lightgbm".to_string(),
            Algorithm::transformers => "transformers".to_string(),
            Algorithm::affinity_propagation => "affinity_propagation".to_string(),
            Algorithm::birch => "birch".to_string(),
            Algorithm::feature_agglomeration => "feature_agglomeration".to_string(),
            Algorithm::mini_batch_kmeans => "mini_batch_kmeans".to_string(),
            Algorithm::mean_shift => "mean_shift".to_string(),
            Algorithm::optics => "optics".to_string(),
            Algorithm::spectral => "spectral".to_string(),
            Algorithm::spectral_bi => "spectral_bi".to_string(),
            Algorithm::spectral_co => "spectral_co".to_string(),
            Algorithm::catboost => "catboost".to_string(),
            Algorithm::pca => "pca".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    /// All algorithm string names for testing
    const ALL_ALGORITHMS: &[(&str, Algorithm)] = &[
        ("linear", Algorithm::linear),
        ("xgboost", Algorithm::xgboost),
        ("xgboost_random_forest", Algorithm::xgboost_random_forest),
        ("svm", Algorithm::svm),
        ("lasso", Algorithm::lasso),
        ("elastic_net", Algorithm::elastic_net),
        ("ridge", Algorithm::ridge),
        ("kmeans", Algorithm::kmeans),
        ("dbscan", Algorithm::dbscan),
        ("knn", Algorithm::knn),
        ("random_forest", Algorithm::random_forest),
        ("least_angle", Algorithm::least_angle),
        ("lasso_least_angle", Algorithm::lasso_least_angle),
        ("orthogonal_matching_pursuit", Algorithm::orthogonal_matching_pursuit),
        ("bayesian_ridge", Algorithm::bayesian_ridge),
        ("automatic_relevance_determination", Algorithm::automatic_relevance_determination),
        ("stochastic_gradient_descent", Algorithm::stochastic_gradient_descent),
        ("perceptron", Algorithm::perceptron),
        ("passive_aggressive", Algorithm::passive_aggressive),
        ("ransac", Algorithm::ransac),
        ("theil_sen", Algorithm::theil_sen),
        ("huber", Algorithm::huber),
        ("quantile", Algorithm::quantile),
        ("kernel_ridge", Algorithm::kernel_ridge),
        ("gaussian_process", Algorithm::gaussian_process),
        ("nu_svm", Algorithm::nu_svm),
        ("ada_boost", Algorithm::ada_boost),
        ("bagging", Algorithm::bagging),
        ("extra_trees", Algorithm::extra_trees),
        ("gradient_boosting_trees", Algorithm::gradient_boosting_trees),
        ("hist_gradient_boosting", Algorithm::hist_gradient_boosting),
        ("linear_svm", Algorithm::linear_svm),
        ("lightgbm", Algorithm::lightgbm),
        ("transformers", Algorithm::transformers),
        ("affinity_propagation", Algorithm::affinity_propagation),
        ("birch", Algorithm::birch),
        ("feature_agglomeration", Algorithm::feature_agglomeration),
        ("mini_batch_kmeans", Algorithm::mini_batch_kmeans),
        ("mean_shift", Algorithm::mean_shift),
        ("optics", Algorithm::optics),
        ("spectral", Algorithm::spectral),
        ("spectral_bi", Algorithm::spectral_bi),
        ("spectral_co", Algorithm::spectral_co),
        ("catboost", Algorithm::catboost),
        ("pca", Algorithm::pca),
    ];

    #[test]
    fn test_from_str_all_algorithms() {
        for (name, expected) in ALL_ALGORITHMS {
            let result = Algorithm::from_str(name);
            assert!(result.is_ok(), "Failed to parse algorithm: {}", name);
            assert_eq!(result.unwrap(), *expected, "Mismatch for algorithm: {}", name);
        }
    }

    #[test]
    fn test_to_string_all_algorithms() {
        for (expected_name, algorithm) in ALL_ALGORITHMS {
            let result = algorithm.to_string();
            assert_eq!(result, *expected_name, "Mismatch for algorithm: {:?}", algorithm);
        }
    }

    #[test]
    fn test_from_str_roundtrip() {
        for (name, _) in ALL_ALGORITHMS {
            let parsed = Algorithm::from_str(name).unwrap();
            let stringified = parsed.to_string();
            assert_eq!(stringified, *name, "Roundtrip failed for: {}", name);
        }
    }

    #[test]
    fn test_from_str_invalid() {
        let invalid_names = &[
            "invalid",
            "XGBOOST",      // case sensitive
            "Linear",       // case sensitive
            "random-forest", // wrong separator
            "",
            "   ",
            "xgboost ",     // trailing space
            " xgboost",     // leading space
        ];

        for name in invalid_names {
            let result = Algorithm::from_str(name);
            assert!(result.is_err(), "Expected error for invalid algorithm: '{}'", name);
        }
    }

    #[test]
    fn test_algorithm_equality() {
        assert_eq!(Algorithm::xgboost, Algorithm::xgboost);
        assert_ne!(Algorithm::xgboost, Algorithm::lightgbm);
        assert_ne!(Algorithm::linear, Algorithm::linear_svm);
    }

    #[test]
    fn test_algorithm_copy_clone() {
        let algo = Algorithm::random_forest;
        let copied = algo;
        let cloned = algo.clone();
        assert_eq!(algo, copied);
        assert_eq!(algo, cloned);
    }

    #[test]
    fn test_algorithm_debug() {
        let algo = Algorithm::xgboost;
        let debug_str = format!("{:?}", algo);
        assert_eq!(debug_str, "xgboost");
    }

    #[test]
    fn test_all_algorithms_count() {
        // Ensure we're testing all 46 algorithms defined in the enum
        assert_eq!(ALL_ALGORITHMS.len(), 46, "Algorithm count mismatch - update tests if enum changed");
    }
}
