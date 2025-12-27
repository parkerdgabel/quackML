use std::collections::{BTreeSet, HashMap};

use ndarray::{Array2, ArrayView1, Zip};

#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum Average {
    Micro,
    Macro,
    Binary,
}

/// Confusion matrix metrics for a class.
#[derive(Debug)]
pub struct ConfusionMatrixMetrics {
    tp: f32,
    fp: f32,
    fn_: f32,

    #[allow(dead_code)]
    tn: f32,
}

impl ConfusionMatrixMetrics {
    /// Args: TP, FP, FN, TN.
    pub fn new(metrics: (f32, f32, f32, f32)) -> ConfusionMatrixMetrics {
        ConfusionMatrixMetrics {
            tp: metrics.0,
            fp: metrics.1,
            fn_: metrics.2,
            tn: metrics.3,
        }
    }
}

pub struct ConfusionMatrix {
    /// The confusion matrix in its raw form.
    matrix: Array2<f32>,

    /// Predicates calculated using the confusion matrix, indexed by class number.
    metrics: Vec<ConfusionMatrixMetrics>,
}

impl ConfusionMatrix {
    /// Construct a new confusion matrix from the ground truth
    /// and the predictions.
    /// `num_classes` is passed it to ensure that all classes
    /// were present in the test set.
    pub fn new(
        ground_truth: &ArrayView1<usize>,
        y_hat: &ArrayView1<usize>,
        num_classes: usize,
    ) -> ConfusionMatrix {
        // Distinct classes.
        let mut classes = ground_truth.iter().collect::<BTreeSet<_>>();
        classes.extend(&mut y_hat.iter().collect::<BTreeSet<_>>().into_iter());

        if ground_truth.len() != y_hat.len() {
            panic!("Can't compute metrics when the ground truth labels are a different size than the predicted labels. {} != {}", ground_truth.len(), y_hat.len());
        }

        if num_classes != classes.len() {
            panic!("Can't compute metrics when the number of classes in the test set is different than the number of classes in the training set. {} != {}", num_classes, classes.len());
        }

        // Class value = index in the confusion matrix
        // e.g. class value 5 will be index 4 if there are classes 1, 2, 3 and 4 present.
        let indexes = classes
            .iter()
            .enumerate()
            .map(|(a, b)| (**b, a))
            .collect::<HashMap<usize, usize>>();

        let mut matrix = Array2::zeros((num_classes, num_classes));

        for (i, t) in ground_truth.iter().enumerate() {
            let h = y_hat[i];

            matrix[(indexes[t], indexes[&h])] += 1.0;
        }

        let mut metrics = Vec::new();

        // Scikit confusion matrix starts from 1 and goes to 0,
        // ours starts from 0 and goes to 1. No big deal,
        // just flip everything lol.
        if num_classes == 2 {
            let tp = matrix[(1, 1)];
            let fp = matrix[(0, 1)];
            let fn_ = matrix[(1, 0)];
            let tn = matrix[(0, 0)];

            metrics.push(ConfusionMatrixMetrics::new((tp, fp, fn_, tn)));
        } else {
            for class in 0..num_classes {
                let tp = matrix[(class, class)];
                let fp = matrix.row(class).sum() - tp;
                let fn_ = matrix.column(class).sum() - tp;
                let tn = matrix.sum() - tp - fp - fn_;

                metrics.push(ConfusionMatrixMetrics::new((tp, fp, fn_, tn)));
            }
        }

        ConfusionMatrix { matrix, metrics }
    }

    pub fn accuracy(&self) -> f32 {
        let numerator = self.matrix.diag().sum();
        let denominator = self.matrix.sum();

        numerator / denominator
    }

    /// Average recall.
    pub fn recall(&self) -> f32 {
        let recalls = self
            .metrics
            .iter()
            .map(|m| m.tp / (m.tp + m.fn_))
            .collect::<Vec<f32>>();

        recalls.iter().sum::<f32>() / recalls.len() as f32
    }

    /// Average precision.
    pub fn precision(&self) -> f32 {
        let precisions = self
            .metrics
            .iter()
            .map(|m| m.tp / (m.tp + m.fp))
            .collect::<Vec<f32>>();

        precisions.iter().sum::<f32>() / precisions.len() as f32
    }

    pub fn f1(&self, average: Average) -> f32 {
        match average {
            Average::Macro => self.f1_macro(),
            Average::Micro | Average::Binary => self.f1_micro(), // micro = binary if num_classes = 2
        }
    }

    /// Calculate the f1 using micro metrics, i.e. the sum of predicates.
    /// This evaluates the classifier as a whole instead of evaluating it as a sum of individual parts.
    fn f1_micro(&self) -> f32 {
        let tp = self.metrics.iter().map(|m| m.tp).sum::<f32>();
        let fn_ = self.metrics.iter().map(|m| m.fn_).sum::<f32>();
        let fp = self.metrics.iter().map(|m| m.fp).sum::<f32>();

        let recall = tp / (tp + fn_);
        let precision = tp / (tp + fp);

        // We risk NaN in f1_micro when precision + recall == 0, because that indicates that
        // both precision and recall are terrible, and the model is likely broken, so giving
        // a NaN that is incomparable to other more valid scores will prevent incorrect
        // comparisons across deceptively comparable scores.
        2. * ((precision * recall) / (precision + recall))
    }

    /// Calculate f1 using the average of class f1's.
    /// This gives equal opportunity to each class to impact the overall score.
    fn f1_macro(&self) -> f32 {
        let recalls = self
            .metrics
            .iter()
            .map(|m| m.tp / (m.tp + m.fn_))
            .map(|x| if x.is_nan() { 1.0 } else { x })
            // .filter(|&x| !x.is_nan())
            .collect::<Vec<f32>>();
        let precisions = self
            .metrics
            .iter()
            .map(|m| m.tp / (m.tp + m.fp))
            // .filter(|&x| !x.is_nan())
            .collect::<Vec<f32>>();

        let mut f1s = Vec::new();

        for (i, recall) in recalls.iter().enumerate() {
            let precision = precisions[i];
            f1s.push(2. * ((precision * recall) / (precision + recall)));
        }

        f1s.iter().sum::<f32>() / f1s.len() as f32
    }
}

pub fn calculate_r2(y_true: &ArrayView1<f32>, y_pred: &ArrayView1<f32>) -> f32 {
    // Calculate the mean of y_true
    let mean_y_true = y_true.mean().unwrap();

    // Calculate Total Sum of Squares (TSS)
    let tss = y_true
        .iter()
        .map(|&y| (y - mean_y_true).powi(2))
        .sum::<f32>();

    // Calculate Residual Sum of Squares (RSS)
    let rss = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&y, &y_hat)| (y - y_hat).powi(2))
        .sum::<f32>();

    // Calculate R²
    1.0 - (rss / tss)
}
pub fn log_loss(y_true: &ArrayView1<f32>, y_pred: &ArrayView1<f32>, eps: f32) -> f32 {
    let n = y_true.len() as f32;
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&y, &p)| {
            let p = p.clamp(eps, 1.0 - eps); // Avoid log(0)
            y * p.ln() + (1.0 - y) * (1.0 - p).ln()
        })
        .sum::<f32>()
        / -n
}

pub fn roc_auc(y_true: &ArrayView1<bool>, y_score: &ArrayView1<f32>) -> f32 {
    let mut pairs: Vec<_> = Zip::from(y_true)
        .and(y_score)
        .map_collect(|a, b| (a, b))
        .into_iter()
        .collect();
    pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let n_pos = y_true.iter().filter(|&&x| x).count();
    let n_neg = y_true.len() - n_pos;

    let mut tpr = 0.0;
    let mut fpr = 0.0;
    let mut auc = 0.0;
    let mut prev_fpr = 0.0;
    let mut prev_score = &f32::INFINITY;

    for (label, score) in pairs.iter() {
        if *score != prev_score {
            auc += tpr * (fpr - prev_fpr);
            prev_score = *score;
            prev_fpr = fpr;
        }
        if **label {
            tpr += 1.0 / n_pos as f32;
        } else {
            fpr += 1.0 / n_neg as f32;
        }
    }
    auc += tpr * (1.0 - prev_fpr);

    auc
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_confusion_matrix_multiclass_perfect() {
        let ground_truth = array![1, 2, 3, 4, 4];
        let y_hat = array![1, 2, 3, 4, 4];

        let mat = ConfusionMatrix::new(
            &ArrayView1::from(&ground_truth),
            &ArrayView1::from(&y_hat),
            4,
        );

        let f1 = mat.f1(Average::Macro);
        let f1_micro = mat.f1(Average::Micro);

        assert_eq!(mat.matrix[(3, 3)], 2.0);
        assert_eq!(f1, 1.0);
        assert_eq!(f1_micro, 1.0);
    }

    #[test]
    fn test_confusion_matrix_binary_perfect() {
        let ground_truth = array![0, 0, 1, 1, 1];
        let y_hat = array![0, 0, 1, 1, 1];

        let mat = ConfusionMatrix::new(
            &ArrayView1::from(&ground_truth),
            &ArrayView1::from(&y_hat),
            2,
        );

        assert_eq!(mat.accuracy(), 1.0);
        assert_eq!(mat.precision(), 1.0);
        assert_eq!(mat.recall(), 1.0);
        assert_eq!(mat.f1(Average::Binary), 1.0);
        assert_eq!(mat.f1(Average::Micro), 1.0);
    }

    #[test]
    fn test_confusion_matrix_binary_imperfect() {
        // Ground truth: [0, 0, 1, 1, 1]
        // Predictions:  [0, 1, 0, 1, 1]
        // TP=2, FP=1, FN=1, TN=1
        let ground_truth = array![0, 0, 1, 1, 1];
        let y_hat = array![0, 1, 0, 1, 1];

        let mat = ConfusionMatrix::new(
            &ArrayView1::from(&ground_truth),
            &ArrayView1::from(&y_hat),
            2,
        );

        // Accuracy = (TP + TN) / total = (2 + 1) / 5 = 0.6
        assert!((mat.accuracy() - 0.6).abs() < 1e-6);

        // Precision = TP / (TP + FP) = 2 / (2 + 1) = 0.666...
        assert!((mat.precision() - 2.0 / 3.0).abs() < 1e-6);

        // Recall = TP / (TP + FN) = 2 / (2 + 1) = 0.666...
        assert!((mat.recall() - 2.0 / 3.0).abs() < 1e-6);

        // F1 = 2 * (precision * recall) / (precision + recall)
        let expected_f1 = 2.0 * (2.0 / 3.0) * (2.0 / 3.0) / (2.0 / 3.0 + 2.0 / 3.0);
        assert!((mat.f1(Average::Binary) - expected_f1).abs() < 1e-6);
    }

    #[test]
    fn test_confusion_matrix_multiclass_imperfect() {
        // 3 classes: 0, 1, 2
        // Ground truth: [0, 0, 1, 1, 2, 2]
        // Predictions:  [0, 1, 1, 2, 2, 0]  (3 correct, 3 wrong)
        let ground_truth = array![0, 0, 1, 1, 2, 2];
        let y_hat = array![0, 1, 1, 2, 2, 0];

        let mat = ConfusionMatrix::new(
            &ArrayView1::from(&ground_truth),
            &ArrayView1::from(&y_hat),
            3,
        );

        // Accuracy = 3/6 = 0.5
        assert!((mat.accuracy() - 0.5).abs() < 1e-6);

        // F1 micro and macro should be <= 1.0 and >= 0.0
        let f1_micro = mat.f1(Average::Micro);
        let f1_macro = mat.f1(Average::Macro);

        assert!(f1_micro >= 0.0 && f1_micro <= 1.0);
        assert!(f1_macro >= 0.0 && f1_macro <= 1.0);
    }

    #[test]
    fn test_confusion_matrix_all_wrong() {
        // All predictions are wrong
        let ground_truth = array![0, 0, 1, 1];
        let y_hat = array![1, 1, 0, 0];

        let mat = ConfusionMatrix::new(
            &ArrayView1::from(&ground_truth),
            &ArrayView1::from(&y_hat),
            2,
        );

        assert_eq!(mat.accuracy(), 0.0);
        // Precision and recall are 0 when there are no true positives
        assert_eq!(mat.precision(), 0.0);
        assert_eq!(mat.recall(), 0.0);
    }

    #[test]
    fn test_calculate_r2_perfect() {
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let r2 = calculate_r2(&y_true.view(), &y_pred.view());
        assert!((r2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_r2_good_fit() {
        // y = 2x, predictions close to actual
        let y_true = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let y_pred = array![2.1, 3.9, 6.2, 7.8, 10.1];

        let r2 = calculate_r2(&y_true.view(), &y_pred.view());
        // R2 should be close to 1 for good predictions
        assert!(r2 > 0.95);
    }

    #[test]
    fn test_calculate_r2_poor_fit() {
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        // Predictions are just the mean (3.0) - should give R2 = 0
        let y_pred = array![3.0, 3.0, 3.0, 3.0, 3.0];

        let r2 = calculate_r2(&y_true.view(), &y_pred.view());
        assert!((r2 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_calculate_r2_negative() {
        // Predictions worse than mean - R2 can be negative
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![5.0, 4.0, 3.0, 2.0, 1.0]; // Inverted predictions

        let r2 = calculate_r2(&y_true.view(), &y_pred.view());
        assert!(r2 < 0.0);
    }

    #[test]
    fn test_log_loss_perfect_predictions() {
        let y_true = array![1.0, 0.0, 1.0, 0.0];
        let y_pred = array![0.99, 0.01, 0.99, 0.01]; // Near-perfect predictions

        let loss = log_loss(&y_true.view(), &y_pred.view(), 1e-15);
        assert!(loss < 0.1); // Should be very low
    }

    #[test]
    fn test_log_loss_poor_predictions() {
        let y_true = array![1.0, 0.0, 1.0, 0.0];
        let y_pred = array![0.5, 0.5, 0.5, 0.5]; // Random guessing

        let loss = log_loss(&y_true.view(), &y_pred.view(), 1e-15);
        // Log loss for random guessing should be around -ln(0.5) ≈ 0.693
        assert!((loss - 0.693).abs() < 0.01);
    }

    #[test]
    fn test_log_loss_inverted_predictions() {
        let y_true = array![1.0, 0.0, 1.0, 0.0];
        let y_pred = array![0.01, 0.99, 0.01, 0.99]; // Inverted predictions

        let loss = log_loss(&y_true.view(), &y_pred.view(), 1e-15);
        assert!(loss > 2.0); // Should be very high
    }

    #[test]
    fn test_roc_auc_perfect() {
        let y_true = array![true, true, false, false];
        let y_score = array![0.9, 0.8, 0.3, 0.1]; // Perfect ranking

        let auc = roc_auc(&y_true.view(), &y_score.view());
        assert!((auc - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_roc_auc_random() {
        // Random predictions should give AUC around 0.5
        let y_true = array![true, false, true, false, true, false];
        let y_score = array![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

        let auc = roc_auc(&y_true.view(), &y_score.view());
        assert!((auc - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_roc_auc_inverted() {
        let y_true = array![true, true, false, false];
        let y_score = array![0.1, 0.2, 0.8, 0.9]; // Inverted ranking

        let auc = roc_auc(&y_true.view(), &y_score.view());
        assert!(auc < 0.1); // Should be close to 0
    }

    #[test]
    fn test_average_enum_equality() {
        assert_eq!(Average::Micro, Average::Micro);
        assert_eq!(Average::Macro, Average::Macro);
        assert_eq!(Average::Binary, Average::Binary);
        assert_ne!(Average::Micro, Average::Macro);
    }

    #[test]
    fn test_confusion_matrix_metrics_new() {
        // TP=10, FP=5, FN=3, TN=82
        let metrics = ConfusionMatrixMetrics::new((10.0, 5.0, 3.0, 82.0));
        assert_eq!(metrics.tp, 10.0);
        assert_eq!(metrics.fp, 5.0);
        assert_eq!(metrics.fn_, 3.0);
        assert_eq!(metrics.tn, 82.0);
    }
}
