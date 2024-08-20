use std::fmt::{Display, Formatter};
use crate::context::DATABASE_CONTEXT;

#[derive(Debug)]
pub struct Dataset {
    pub x_train: Vec<f32>,
    pub y_train: Vec<f32>,
    pub x_test: Vec<f32>,
    pub y_test: Vec<f32>,
    pub num_features: usize,
    pub num_labels: usize,
    pub num_rows: usize,
    pub num_train_rows: usize,
    pub num_test_rows: usize,
    pub num_distinct_labels: usize,
}

impl Display for Dataset {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "Dataset {{ num_features: {}, num_labels: {}, num_distinct_labels: {}, num_rows: {}, num_train_rows: {}, num_test_rows: {} }}",
            self.num_features, self.num_labels, self.num_distinct_labels, self.num_rows, self.num_train_rows, self.num_test_rows,
        )
    }
}


impl Dataset {
    pub fn fold(&self, k: usize, folds: usize) -> Dataset {
        if folds < 2 {
            panic!("Number of folds must be at least 2");
        }
        let fold_test_size = self.num_train_rows / folds;
        let test_start = k * fold_test_size;
        let test_end = test_start + fold_test_size;
        let num_train_rows = self.num_train_rows - fold_test_size;

        let x_test_start = test_start * self.num_features;
        let x_test_end = test_end * self.num_features;
        let y_test_start = test_start * self.num_labels;
        let y_test_end = test_end * self.num_labels;

        let mut x_train = Vec::with_capacity(num_train_rows * self.num_features);
        x_train.extend_from_slice(&self.x_train[..x_test_start]);
        x_train.extend_from_slice(&self.x_train[x_test_end..]);
        let mut y_train = Vec::with_capacity(num_train_rows * self.num_labels);
        y_train.extend_from_slice(&self.y_train[..y_test_start]);
        y_train.extend_from_slice(&self.y_train[y_test_end..]);

        let x_test = self.x_train[x_test_start..x_test_end].to_vec();
        let y_test = self.y_train[y_test_start..y_test_end].to_vec();

        Dataset {
            x_train,
            y_train,
            x_test,
            y_test,
            num_features: self.num_features,
            num_labels: self.num_labels,
            num_rows: self.num_train_rows,
            num_train_rows,
            num_test_rows: fold_test_size,
            num_distinct_labels: self.num_distinct_labels,
        }
    }
}

pub enum TextDatasetType {
    TextClassification(TextClassificationDataset),
    TextPairClassification(TextPairClassificationDataset),
    Conversation(ConversationDataset),
}

impl TextDatasetType {
    pub fn num_features(&self) -> usize {
        match self {
            TextDatasetType::TextClassification(dataset) => dataset.num_features,
            TextDatasetType::TextPairClassification(dataset) => dataset.num_features,
            TextDatasetType::Conversation(dataset) => dataset.num_features,
        }
    }
}


// TextClassificationDataset
pub struct TextClassificationDataset {
    pub text_train: Vec<String>,
    pub class_train: Vec<String>,
    pub text_test: Vec<String>,
    pub class_test: Vec<String>,
    pub num_features: usize,
    pub num_labels: usize,
    pub num_rows: usize,
    pub num_train_rows: usize,
    pub num_test_rows: usize,
    pub num_distinct_labels: usize,
}

impl Display for TextClassificationDataset {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "TextClassificationDataset {{ num_distinct_labels: {}, num_rows: {}, num_train_rows: {}, num_test_rows: {} }}",
            self.num_distinct_labels, self.num_rows, self.num_train_rows, self.num_test_rows,
        )
    }
}

pub struct TextPairClassificationDataset {
    pub text1_train: Vec<String>,
    pub text2_train: Vec<String>,
    pub class_train: Vec<String>,
    pub text1_test: Vec<String>,
    pub text2_test: Vec<String>,
    pub class_test: Vec<String>,
    pub num_features: usize,
    pub num_labels: usize,
    pub num_rows: usize,
    pub num_train_rows: usize,
    pub num_test_rows: usize,
    pub num_distinct_labels: usize,
}

impl Display for TextPairClassificationDataset {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "TextPairClassificationDataset {{ num_distinct_labels: {}, num_rows: {}, num_train_rows: {}, num_test_rows: {} }}",
            self.num_distinct_labels, self.num_rows, self.num_train_rows, self.num_test_rows,
        )
    }
}

pub struct ConversationDataset {
    pub system_train: Vec<String>,
    pub user_train: Vec<String>,
    pub assistant_train: Vec<String>,
    pub system_test: Vec<String>,
    pub user_test: Vec<String>,
    pub assistant_test: Vec<String>,
    pub num_features: usize,
    pub num_rows: usize,
    pub num_train_rows: usize,
    pub num_test_rows: usize,
}

impl Display for ConversationDataset {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "ConversationDataset {{ num_rows: {}, num_train_rows: {}, num_test_rows: {} }}",
            self.num_rows, self.num_train_rows, self.num_test_rows,
        )
    }
}

fn drop_table_if_exists(table_name: &str) {
    let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
    // Avoid the existence for DROP TABLE IF EXISTS warning by checking the schema for the table first
    let table_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = $1 AND table_schema = 'quackml'",
        [table_name],
        |row| row.get(0))
    .unwrap();
    if table_count == 1 {
        // Drop the table if it exists
        conn.execute_batch(&format!("DROP TABLE IF EXISTS quackml.{}", table_name)).unwrap();
    }
}
