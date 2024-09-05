use crate::context::DATABASE_CONTEXT;
use crate::orm::Sampling;
use crate::orm::Status;
use crate::orm::TextClassificationDataset;
use duckdb::params;
use duckdb::Row;
use serde_with::serde_as;
use serde_with::DefaultOnNull;
use duckdb::OptionalExt;
use indexmap::IndexMap;
use ndarray::Zip;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::{Display, Error, Formatter};
use std::str::FromStr;

use super::ConversationDataset;
use super::Dataset;
use super::TextPairClassificationDataset;

// Categories use a designated string to represent NULL categorical values,
// rather than Option<String> = None, because the JSONB serialization schema
// only supports String keys in JSON objects, unlike a Rust HashMap.
pub(crate) const NULL_CATEGORY_KEY: &str = "__NULL__";

// A category maintains the encoded value for a key, as well as counting the number
// of members in the training set for statistical purposes, e.g. target_mean
#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub(crate) struct Category {
    pub(crate) value: f32,
    pub(crate) members: usize,
}

// Statistics are computed for every column over the training data when the
// data is read.
// TODO: This deserialize_as is wrong. Need to write a function to deserialize into NAN
#[serde_as]
#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub(crate) struct Statistics {
    #[serde_as(deserialize_as = "DefaultOnNull")]
    min: f32,
    #[serde_as(deserialize_as = "DefaultOnNull")]
    max: f32,
    #[serde_as(deserialize_as = "DefaultOnNull")]
    max_abs: f32,
    #[serde_as(deserialize_as = "DefaultOnNull")]
    mean: f32,
    #[serde_as(deserialize_as = "DefaultOnNull")]
    median: f32,
    #[serde_as(deserialize_as = "DefaultOnNull")]
    mode: f32,
    #[serde_as(deserialize_as = "DefaultOnNull")]
    variance: f32,
    #[serde_as(deserialize_as = "DefaultOnNull")]
    std_dev: f32,
    #[serde_as(deserialize_as = "DefaultOnNull")]
    missing: usize,
    #[serde_as(deserialize_as = "DefaultOnNull")]
    distinct: usize,
    #[serde_as(deserialize_as = "DefaultOnNull")]
    histogram: Vec<usize>,
    #[serde_as(deserialize_as = "Vec<DefaultOnNull>")]
    ventiles: Vec<f32>,
    pub categories: Option<HashMap<String, Category>>,
}

impl Default for Statistics {
    fn default() -> Self {
        Statistics {
            min: f32::NAN,
            max: f32::NAN,
            max_abs: f32::NAN,
            mean: f32::NAN,
            median: f32::NAN,
            mode: f32::NAN,
            variance: f32::NAN,
            std_dev: f32::NAN,
            missing: 0,
            distinct: 0,
            histogram: vec![0; 20],
            ventiles: vec![f32::NAN; 19],
            categories: None,
        }
    }
}

// How to encode categorical values
// TODO add limit and min_frequency params to all
#[derive(Debug, Default, PartialEq, Serialize, Deserialize, Clone)]
#[allow(non_camel_case_types)]
pub(crate) enum Encode {
    // For use with algorithms that directly support the data type
    #[default]
    native,
    // Encode each category as the mean of the target
    target,
    // Encode each category as one boolean column per category
    one_hot,
    // Encode each category as ascending integer values
    ordinal(Vec<String>),
}

// How to replace missing values
#[derive(Debug, Default, PartialEq, Serialize, Deserialize, Clone)]
#[allow(non_camel_case_types)]
pub(crate) enum Impute {
    #[default]
    // Raises an panic at runtime
    panic,
    mean,
    median,
    mode,
    min,
    max,
    // Replaces with 0
    zero,
}

#[derive(Debug, Default, PartialEq, Serialize, Deserialize, Clone)]
#[allow(non_camel_case_types)]
pub(crate) enum Scale {
    #[default]
    preserve,
    standard,
    min_max,
    max_abs,
    robust,
}

#[derive(Debug, Default, PartialEq, Serialize, Deserialize, Clone)]
pub(crate) struct Preprocessor {
    #[serde(default)]
    encode: Encode,
    #[serde(default)]
    impute: Impute,
    #[serde(default)]
    scale: Scale,
}

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
pub struct Column {
    pub(crate) name: String,
    pub(crate) duckdb_type: String,
    pub(crate) nullable: bool,
    pub(crate) label: bool,
    pub(crate) position: usize,
    pub(crate) size: usize,
    pub(crate) array: bool,
    #[serde(default)]
    pub(crate) preprocessor: Preprocessor,
    #[serde(default)]
    pub(crate) statistics: Statistics,
}

impl Column {
    fn categorical_type(duckdb_type: &str) -> bool {
        Column::nominal_type(duckdb_type) || Column::ordinal_type(duckdb_type)
    }

    fn ordinal_type(_duckdb_type: &str) -> bool {
        false
    }

    fn nominal_type(duckdb_type: &str) -> bool {
        matches!(
            duckdb_type,
            "bpchar" | "text" | "varchar" | "bpchar[]" | "text[]" | "varchar[]"
        )
    }

    pub(crate) fn quoted_name(&self) -> String {
        format!(r#""{}""#, self.name)
    }

    #[inline]
    pub(crate) fn get_category_value(&self, key: &str) -> f32 {
        match self.statistics.categories.as_ref().unwrap().get(key) {
            Some(category) => category.value,
            None => f32::NAN,
        }
    }

    #[inline]
    pub(crate) fn scale(&self, value: f32) -> f32 {
        match self.preprocessor.scale {
            Scale::standard => (value - self.statistics.mean) / self.statistics.std_dev,
            Scale::min_max => {
                (value - self.statistics.min) / (self.statistics.max - self.statistics.min)
            }
            Scale::max_abs => value / self.statistics.max_abs,
            Scale::robust => {
                (value - self.statistics.median)
                    / (self.statistics.ventiles[15] - self.statistics.ventiles[5])
            }
            Scale::preserve => value,
        }
    }

    #[inline]
    pub(crate) fn impute(&self, value: f32) -> f32 {
        if value.is_nan() {
            match &self.preprocessor.impute {
                Impute::mean => self.statistics.mean,
                Impute::median => self.statistics.median,
                Impute::mode => self.statistics.mode,
                Impute::min => self.statistics.min,
                Impute::max => self.statistics.max,
                Impute::zero => 0.,
                Impute::panic => panic!("{} missing values for {}. You may provide a preprocessor to impute a value. e.g:\n\n pgml.train(preprocessor => '{{{:?}: {{\"impute\": \"mean\"}}}}'", self.statistics.missing, self.name, self.name),
            }
        } else {
            value
        }
    }

    pub(crate) fn encoded_width(&self) -> usize {
        match self.preprocessor.encode {
            Encode::one_hot => self.statistics.categories.as_ref().unwrap().len() - 1,
            _ => 1,
        }
    }

    pub(crate) fn array_width(&self) -> usize {
        self.size
    }

    pub(crate) fn preprocess(
        &self,
        data: &ndarray::ArrayView<f32, ndarray::Ix1>,
        processed_data: &mut [f32],
        features_width: usize,
        position: usize,
    ) {
        for (row, &d) in data.iter().enumerate() {
            let value = self.impute(d);
            match &self.preprocessor.encode {
                Encode::one_hot => {
                    for i in 0..self.statistics.categories.as_ref().unwrap().len() - 1 {
                        let one_hot = if i == value as usize { 1. } else { 0. } as f32;
                        processed_data[row * features_width + position + i] = one_hot;
                    }
                }
                _ => processed_data[row * features_width + position] = self.scale(value),
            };
        }
    }

    fn analyze(
        &mut self,
        array: &ndarray::ArrayView<f32, ndarray::Ix1>,
        target: &ndarray::ArrayView<f32, ndarray::Ix1>,
    ) {
        // target encode if necessary before analyzing
        if self.preprocessor.encode == Encode::target {
            let categories = self.statistics.categories.as_mut().unwrap();
            let mut sums = vec![0_f32; categories.len() + 1];
            let mut total = 0.;
            Zip::from(array).and(target).for_each(|&value, &target| {
                total += target;
                sums[value as usize] += target;
            });
            let avg_target = total / categories.len() as f32;
            for category in categories.values_mut() {
                if category.members > 0 {
                    let sum = sums[category.value as usize];
                    category.value = sum / category.members as f32;
                } else {
                    // use avg target for categories w/ no members, e.g. __NULL__ category in a complete dataset
                    category.value = avg_target;
                }
            }
        }

        // Data is filtered for NaN because it is not well-defined statistically, and they are counted as separate stat
        let mut data = array
            .iter()
            .filter_map(|n| if n.is_nan() { None } else { Some(*n) })
            .collect::<Vec<f32>>();
        data.sort_by(|a, b| a.total_cmp(b));

        // FixMe: Arrays are analyzed many times, clobbering/appending to the same stats, columns are also re-analyzed in memory during tests, which can cause unnexpected failures
        let statistics = &mut self.statistics;
        statistics.min = *data.first().unwrap();
        statistics.max = *data.last().unwrap();
        statistics.max_abs = if statistics.min.abs() > statistics.max.abs() {
            statistics.min.abs()
        } else {
            statistics.max.abs()
        };
        statistics.mean = data.iter().sum::<f32>() / data.len() as f32;
        statistics.median = data[data.len() / 2];
        statistics.missing = array.len() - data.len();
        if self.label && statistics.missing > 0 {
            panic!("The training data labels in \"{}\" contain {} NULL values. Consider filtering these values from the training data by creating a VIEW that includes a SQL filter like `WHERE {} IS NOT NULL`.", self.name, statistics.missing, self.name);
        }
        statistics.variance = data
            .iter()
            .map(|i| {
                let diff = statistics.mean - (*i);
                diff * diff
            })
            .sum::<f32>()
            / data.len() as f32;
        statistics.std_dev = statistics.variance.sqrt();
        let histogram_boundaries = ndarray::Array::linspace(statistics.min, statistics.max, 21);
        let mut h = 0;
        let ventile_size = data.len() as f32 / 20.;
        let mut streak = 1;
        let mut max_streak = 0;
        let mut previous = f32::NAN;

        let mut modes = Vec::new();
        statistics.distinct = 0; // necessary reset before array columns clobber
        for (i, &value) in data.iter().enumerate() {
            // mode candidates form streaks
            if value == previous {
                streak += 1;
            } else if !previous.is_nan() {
                match streak.cmp(&max_streak) {
                    Ordering::Greater => {
                        modes = vec![previous];
                        max_streak = streak;
                    }
                    Ordering::Equal => modes.push(previous),
                    Ordering::Less => {}
                }
                streak = 1;
                statistics.distinct += 1;
            }
            previous = value;

            // histogram
            while value >= histogram_boundaries[h] && h < statistics.histogram.len() {
                h += 1;
            }
            statistics.histogram[h - 1] += 1;

            // ventiles
            let v = (i as f32 / ventile_size) as usize;
            if v < 19 {
                statistics.ventiles[v] = value;
            }
        }
        // Pick the mode in the middle of all the candidates with the longest streaks
        if !previous.is_nan() {
            statistics.distinct += 1;
            if streak > max_streak {
                statistics.mode = previous;
            } else {
                statistics.mode = modes[modes.len() / 2];
            }
        }

        // Fill missing ventiles with the preceding value, when there are fewer than 20 points
        for i in 1..statistics.ventiles.len() {
            if statistics.ventiles[i].is_nan() {
                statistics.ventiles[i] = statistics.ventiles[i - 1];
            }
        }

        println!("Column {:?}: {:?}", self.name, statistics);
    }
}

impl PartialOrd<Self> for Column {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Column {}

impl Ord for Column {
    fn cmp(&self, other: &Self) -> Ordering {
        self.position.cmp(&other.position)
    }
}

// Array and one hot encoded columns take up multiple positions in a feature row
#[derive(Debug, Clone)]
pub struct ColumnRowPosition {
    pub(crate) column_position: usize,
    pub(crate) row_position: usize,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Snapshot {
    pub(crate) id: i64,
    pub(crate) relation_name: String,
    pub(crate) y_column_name: String,
    pub(crate) test_size: f32,
    pub(crate) test_sampling: Sampling,
    pub(crate) status: Status,
    pub(crate) columns: Vec<Column>,
    pub(crate) analysis: Option<IndexMap<String, f32>>,
    pub(crate) created_at: String,
    pub(crate) updated_at: String,
    pub(crate) materialized: bool,
    pub(crate) feature_positions: Vec<ColumnRowPosition>,
}

impl Display for Snapshot {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "Snapshot {{ id: {}, relation_name: {}, y_column_name: {:?}, test_size: {}, test_sampling: {:?}, status: {:?} }}", self.id, self.relation_name, self.y_column_name, self.test_size, self.test_sampling, self.status)
    }
}

impl Snapshot {
    fn from_row(row: &Row) -> Snapshot {
        let json: String = row.get(6).unwrap();
        let columns: Vec<Column> = serde_json::from_str(&json).unwrap();
        let result = row.get(7);
        let json = match result {
            Ok(s) => s,
            Err(_) => "{}".to_string()
        };
        let analysis: Option<IndexMap<String, f32>> = Some(serde_json::from_str(&json).unwrap());
        let y_column_name = row.get::<_, String>(2).unwrap();
        
        let mut s = Snapshot {
            id: row.get(0).unwrap(),
            relation_name: row.get::<_, String>(1).unwrap(),
            y_column_name,
            test_size: row.get(3).unwrap(),
            test_sampling: Sampling::from_str(&row.get::<_, String>(4).unwrap()).unwrap(),
            status: Status::from_str(&row.get::<_, String>(5).unwrap()).unwrap(),
            columns,
            analysis,
            created_at: row.get(8).unwrap(),
            updated_at: row.get(9).unwrap(),
            materialized: row.get(10).unwrap(),
            feature_positions: Vec::new(),
        };
        s.feature_positions = s.feature_positions();
        s
    }
    pub fn find(id: i64) -> Option<Snapshot> {
        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
        conn.query_row_and_then(
            "SELECT
                        snapshots.id,
                        snapshots.relation_name,
                        snapshots.y_column_name,
                        snapshots.test_size,
                        snapshots.test_sampling::TEXT,
                        snapshots.status::TEXT,
                        snapshots.columns,
                        snapshots.analysis,
                        snapshots.created_at,
                        snapshots.updated_at,
                        snapshots.materialized
                    FROM quackml.snapshots
                    WHERE id = $1
                    ORDER BY snapshots.id DESC
                    LIMIT 1;
                    ",
            [id],
            |row| {
                let s = Snapshot::from_row(row);
                Ok(s)
            },
        )
        .optional()
        .unwrap()
    }

    pub fn find_last_by_project_id(project_id: i64) -> Option<Snapshot> {
        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
        conn.query_row_and_then(
            "SELECT
                snapshots.id,
                snapshots.relation_name,
                snapshots.y_column_name,
                snapshots.test_size,
                snapshots.test_sampling::TEXT,
                snapshots.status::TEXT,
                snapshots.columns,
                snapshots.analysis,
                snapshots.created_at,
                snapshots.updated_at,
                snapshots.materialized
            FROM quackml.snapshots
            JOIN quackml.models
              ON models.snapshot_id = snapshots.id
              AND models.project_id = $1
            ORDER BY snapshots.id DESC
            LIMIT 1;
            ",
            [project_id],
            |row| {
                let s = Snapshot::from_row(row);
                Ok(s)
            },
        )
        .optional()
        .unwrap()
    }

    fn get_columns(
        schema_name: &str,
        table_name: &str,
        preprocessors: HashMap<String, Preprocessor>,
        y_column_name: Option<String>,
    ) -> Vec<Column> {
        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
        let mut binding = conn
        .prepare("SELECT column_name::TEXT, data_type::TEXT, is_nullable FROM information_schema.columns WHERE table_schema = $1 AND table_name = $2 ORDER BY ordinal_position ASC")
        .unwrap();
        let res  =binding
        .query_and_then(
            params![schema_name, table_name],
            |row| -> Result<Column, duckdb::Error> {
                let mut position = 0;
                let name = row.get::<_, String>(0).unwrap();
                let mut duckdb_type = row.get::<_, String>(1).unwrap();
                let mut size = 1;
                let mut array = false;
                if duckdb_type.starts_with('_') {
                    size = 0;
                    array = true;
                    duckdb_type = duckdb_type[1..].to_string() + "[]";
                }
                let nullable = row.get::<_, String>(2).map(|s| match s.as_str() {
                        "YES" => true,
                        "NO" => false,
                        _ => panic!("Expected YES or NO"),
                                           
                    }).unwrap();
                position += 1;
                let label = match y_column_name {
                    Some(ref y_column_name) => y_column_name.eq(&name),
                    None => false,
                };
                let mut statistics = Statistics::default();
                let preprocessor = match preprocessors.get(&name) {
                    Some(preprocessor) => {
                        let preprocessor = preprocessor.clone();
                        if Column::categorical_type(&duckdb_type) {
                            if preprocessor.impute == Impute::mean && preprocessor.encode != Encode::target {
                                panic!("panic initializing preprocessor for column: {:?}.\n\n  You can not specify {{\"impute: mean\"}} for a categorical variable unless it is also encoded using `target_mean`, because there is no \"average\" category. `{{\"impute: mode\"}}` is valid alternative, since there is a most common category. Another option would be to encode using target_mean, and then the target mean will be imputed for missing categoricals.", name);
                            }
                        } else if preprocessor.encode != Encode::native {
                            panic!("panic initializing preprocessor for column: {:?}.\n\n  It does not make sense to {{\"encode: {:?}}} a continuous variable. Please use the default `native`.", name, preprocessor.scale);
                        }
                        preprocessor
                    },
                    None => Preprocessor::default(),
                };

                if Column::categorical_type(&duckdb_type) || preprocessor.encode != Encode::native {
                    let mut categories = HashMap::new();
                    categories.insert(
                        NULL_CATEGORY_KEY.to_string(),
                        Category {
                            value: 0.,
                            members: 0,
                        }
                    );
                    statistics.categories = Some(categories);
                }
                Ok(Column {
                    name,
                    duckdb_type,
                    nullable,
                    label,
                    position,
                    size,
                    array,
                    statistics,
                    preprocessor,
                })
            }
        );
        let mut cols = Vec::new();
        for row in res.unwrap().into_iter() {
            cols.push(row.unwrap());
        }
        cols
    }

    pub fn create(
        relation_name: &str,
        y_column_name: Option<String>,
        test_size: f32,
        test_sampling: Sampling,
        materialized: bool,
        preprocess: &str,
    ) -> Snapshot {
        let status = Status::in_progress;

        // Validate table exists.
        let (schema_name, table_name) = Self::fully_qualified_table(relation_name);

        let val: Value = serde_json::from_str(preprocess).unwrap();
        let preprocessors: HashMap<String, Preprocessor> =
            val.as_object().map_or(HashMap::new(), |m| {
                m.iter()
                    .map(|(k, v)| (k.clone(), serde_json::from_value(v.clone()).unwrap()))
                    .collect()
            });

        let columns = Self::get_columns(
            &schema_name,
            &table_name,
            preprocessors.clone(),
            y_column_name.clone(),
        );

        if let Some(y_column_name) = &y_column_name {
            if !columns.iter().any(|c| c.label && &c.name == y_column_name) {
                panic!(
                    "Column `{}` not found. Did you pass the correct `y_column_name`?",
                    y_column_name
                )
            }
        }
        let y_column_name_param = match y_column_name {
            Some(y_column_name) => duckdb::types::Value::Text(y_column_name.clone()),
            None => duckdb::types::Value::Null,
        };
        let columns_param = duckdb::types::Value::Text(serde_json::to_string(&columns).unwrap());
        let test_size_param = duckdb::types::Value::Float(test_size);
        let relation_name_param = duckdb::types::Value::Text(relation_name.to_string());
        let test_sampling_param = duckdb::types::Value::Text(test_sampling.to_string());
        let status_param = duckdb::types::Value::Text(status.to_string());
        let materialzed_param = duckdb::types::Value::Boolean(materialized);
        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
        conn.query_row("INSERT INTO quackml.snapshots (relation_name, y_column_name, test_size, test_sampling, status, columns, materialized) VALUES ($1, $2, $3, $4::sampling, $5, $6, $7) RETURNING id, relation_name, y_column_name, test_size, test_sampling::TEXT, status::TEXT, columns, analysis, created_at, updated_at, materialized;",
        params![relation_name_param, y_column_name_param, test_size_param, test_sampling_param, status_param, columns_param, materialzed_param],
        |row| {
            let s = Snapshot::from_row(&row);

            if materialized {
                println!("Materializing snapshot {}", s.id);
                let sampled_query = s.test_sampling.get_sql(&s.relation_name, s.columns.clone());
                let sql = format!(
                    r#"CREATE TABLE "quackml"."snapshot_{}" AS {}"#,
                    s.id, sampled_query
                );
                conn.execute_batch(&sql).unwrap();
            }
            Ok(s)
        }
    ).unwrap()
    }

    pub(crate) fn labels(&self) -> impl Iterator<Item = &Column> {
        self.columns.iter().filter(|c| c.label)
    }

    pub(crate) fn label_positions(&self) -> Vec<ColumnRowPosition> {
        let mut label_positions = Vec::with_capacity(self.num_labels());
        let mut row_position = 0;
        for column in self.labels() {
            for _ in 0..column.size {
                label_positions.push(ColumnRowPosition {
                    column_position: column.position,
                    row_position,
                });
                row_position += column.encoded_width();
            }
        }
        label_positions
    }

    pub(crate) fn features(&self) -> impl Iterator<Item = &Column> {
        self.columns.iter().filter(|c| !c.label)
    }

    pub(crate) fn feature_positions(&self) -> Vec<ColumnRowPosition> {
        let mut feature_positions = Vec::with_capacity(self.num_features());
        let mut row_position = 0;
        for column in self.features() {
            for _ in 0..column.size {
                feature_positions.push(ColumnRowPosition {
                    column_position: column.position,
                    row_position,
                });
                row_position += column.encoded_width();
            }
        }
        feature_positions
    }

    pub(crate) fn num_labels(&self) -> usize {
        self.labels().map(|f| f.size).sum::<usize>()
    }

    pub(crate) fn first_label(&self) -> &Column {
        self.labels()
            .find(|l| l.name == self.y_column_name)
            .unwrap()
    }

    pub(crate) fn num_classes(&self) -> usize {
        match &self.y_column_name.len() {
            0 => 0,
            _ => match &self.first_label().statistics.categories {
                Some(categories) => categories.len(),
                None => self.first_label().statistics.distinct,
            },
        }
    }

    pub(crate) fn num_features(&self) -> usize {
        self.features().map(|c| c.size).sum::<usize>()
    }

    pub(crate) fn features_width(&self) -> usize {
        self.features()
            .map(|f| f.array_width() * f.encoded_width())
            .sum::<usize>()
    }

    fn fully_qualified_table(relation_name: &str) -> (String, String) {
        let parts = relation_name
            .split('.')
            .map(|name| name.to_string())
            .collect::<Vec<String>>();

        let (schema_name, table_name) = match parts.len() {
            1 => (None, parts[0].clone()),
            2 => (Some(parts[0].clone()), parts[1].clone()),
            _ => panic!(
                "Relation name \"{}\" is not parsable into schema name and table name",
                relation_name
            ),
        };

        match schema_name {
            None => {
                let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
                let table_count = conn.query_row(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = $1 AND table_schema = 'public'",
                    [&table_name],
                    |row| row.get(0),
                ).unwrap();

                let panic = format!("Relation \"{}\" could not be found in the public schema. Please specify the table schema, e.g. pgml.{}", &table_name, &table_name);

                match table_count {
                    0 => panic!("{}", panic),
                    1 => (String::from("public"), table_name),
                    _ => panic!("{}", panic),
                }
            }

            Some(schema_name) => {
                let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
                let exists = conn.query_row(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = $1 AND table_schema = $2",
                    [&table_name, &schema_name],
                    |row| row.get(0),
                ).ok();
                match exists {
                    None => panic!(
                        "Relation \"{}\".\"{}\" doesn't exist",
                        schema_name, table_name
                    ),
                    Some(1) => (schema_name, table_name),
                    _ => panic!(
                        "Relation \"{}\".\"{}\" is ambiguous",
                        schema_name, table_name
                    ),
                }
            }
        }
    }

    fn select_sql(&self) -> String {
        match self.materialized {
            true => {
                format!(
                    "SELECT {} FROM {}",
                    self.columns
                        .iter()
                        .map(|c| c.quoted_name())
                        .collect::<Vec<String>>()
                        .join(", "),
                    self.relation_name_quoted()
                )
            }
            false => self
                .test_sampling
                .get_sql(&self.relation_name_quoted(), self.columns.clone()),
        }
    }

    fn train_test_split(&self, num_rows: usize) -> (usize, usize) {
        let num_test_rows = if self.test_size > 1.0 {
            self.test_size as usize
        } else {
            (num_rows as f32 * self.test_size).round() as usize
        };

        let num_train_rows = num_rows - num_test_rows;
        if num_train_rows == 0 {
            panic!(
                "test_size = {} is too large. There are only {} samples.",
                num_test_rows, num_rows
            );
        }

        (num_train_rows, num_test_rows)
    }

    pub fn text_classification_dataset(
        &mut self,
        dataset_args: &Value,
    ) -> TextClassificationDataset {
        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
        let mut stmt = conn.prepare(&self.select_sql()).unwrap();

        let num_rows = stmt.row_count();

        let (num_train_rows, num_test_rows) = self.train_test_split(num_rows);
        let num_features = self.num_features();
        let num_labels = self.num_labels();

        let mut text_train: Vec<String> = Vec::with_capacity(num_train_rows);
        let mut class_train: Vec<String> = Vec::with_capacity(num_train_rows);
        let mut text_test: Vec<String> = Vec::with_capacity(num_test_rows);
        let mut class_test: Vec<String> = Vec::with_capacity(num_test_rows);

        let class_column_value = dataset_args
            .get("class_column")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "class".to_string());

        let text_column_value = dataset_args
            .get("text_column")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "text".to_string());

        let mut rows = stmt.query([]).unwrap();
        let mut i = 0;
        while let Some(row) = rows.next().unwrap() {
            let row = row;
            for column in &mut self.columns {
                let vector = if column.name == text_column_value {
                    if i < num_train_rows {
                        &mut text_train
                    } else {
                        &mut text_test
                    }
                } else if column.name == class_column_value {
                    if i < num_train_rows {
                        &mut class_train
                    } else {
                        &mut class_test
                    }
                } else {
                    continue;
                };

                match column.duckdb_type.as_str() {
                    "bpchar" | "text" | "varchar" => {
                        match row.get::<_, Option<String>>(column.position).unwrap() {
                            Some(text) => vector.push(text),
                            None => panic!("NULL training text is not handled"),
                        }
                    }
                    _ => panic!("only text type columns are supported"),
                }
                i += 1;
            }
        }
        let num_distinct_labels = class_train.iter().cloned().collect::<HashSet<_>>().len();
        TextClassificationDataset {
            text_train,
            class_train,
            text_test,
            class_test,
            num_features,
            num_labels,
            num_rows,
            num_test_rows,
            num_train_rows,
            num_distinct_labels,
        }
    }

    pub fn text_pair_classification_dataset(
        &mut self,
        dataset_args: &Value,
    ) -> TextPairClassificationDataset {
        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
        let mut stmt = conn.prepare(&self.select_sql()).unwrap();

        let num_rows = stmt.row_count();
        let (num_train_rows, num_test_rows) = self.train_test_split(num_rows);
        let num_features = 2;
        let num_labels = self.num_labels();

        let mut text1_train: Vec<String> = Vec::with_capacity(num_train_rows);
        let mut text2_train: Vec<String> = Vec::with_capacity(num_train_rows);
        let mut class_train: Vec<String> = Vec::with_capacity(num_train_rows);
        let mut text1_test: Vec<String> = Vec::with_capacity(num_test_rows);
        let mut text2_test: Vec<String> = Vec::with_capacity(num_test_rows);
        let mut class_test: Vec<String> = Vec::with_capacity(num_test_rows);

        let text1_column_value = dataset_args
            .get("text1_column")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "text1".to_string());

        let text2_column_value = dataset_args
            .get("text2_column")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "text2".to_string());

        let class_column_value = dataset_args
            .get("class_column")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "class".to_string());

        let mut rows = stmt.query([]).unwrap();
        let mut i = 0;
        while let Some(row) = rows.next().unwrap() {
            let row = row;
            for column in &mut self.columns {
                let vector = if column.name == text1_column_value {
                    if i < num_train_rows {
                        &mut text1_train
                    } else {
                        &mut text1_test
                    }
                } else if column.name == text2_column_value {
                    if i < num_train_rows {
                        &mut text2_train
                    } else {
                        &mut text2_test
                    }
                } else if column.name == class_column_value {
                    if i < num_train_rows {
                        &mut class_train
                    } else {
                        &mut class_test
                    }
                } else {
                    continue;
                };

                match column.duckdb_type.as_str() {
                    "bpchar" | "text" | "varchar" => {
                        match row.get::<_, Option<String>>(column.position).unwrap() {
                            Some(text) => vector.push(text),
                            None => panic!("NULL training text is not handled"),
                        }
                    }
                    _ => panic!("only text type columns are supported"),
                }
            }
            i += 1;
        }

        let num_distinct_labels = class_train.iter().cloned().collect::<HashSet<_>>().len();

        TextPairClassificationDataset {
            text1_train,
            text2_train,
            class_train,
            text1_test,
            text2_test,
            class_test,
            num_features,
            num_labels,
            num_rows,
            num_test_rows,
            num_train_rows,
            num_distinct_labels,
        }
    }

    pub fn conversation_dataset(&mut self, dataset_args: &Value) -> ConversationDataset {
        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
        let mut stmt = conn.prepare(&self.select_sql()).unwrap();
        let num_rows = stmt.row_count();
        let (num_train_rows, num_test_rows) = self.train_test_split(num_rows);
        let num_features = 2;

        let mut system_train: Vec<String> = Vec::with_capacity(num_train_rows);
        let mut user_train: Vec<String> = Vec::with_capacity(num_train_rows);
        let mut assistant_train: Vec<String> = Vec::with_capacity(num_train_rows);
        let mut system_test: Vec<String> = Vec::with_capacity(num_test_rows);
        let mut user_test: Vec<String> = Vec::with_capacity(num_test_rows);
        let mut assistant_test: Vec<String> = Vec::with_capacity(num_test_rows);

        let system_column_value = dataset_args
            .get("system_column")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "system".to_string());

        let user_column_value = dataset_args
            .get("user_column")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "user".to_string());

        let assistant_column_value = dataset_args
            .get("assistant_column")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "assistant".to_string());

        let mut rows = stmt.query([]).unwrap();
        let mut i = 0;
        while let Some(row) = rows.next().unwrap() {
            let row = row;
            for column in &mut self.columns {
                let vector = if column.name == system_column_value {
                    if i < num_train_rows {
                        &mut system_train
                    } else {
                        &mut system_test
                    }
                } else if column.name == user_column_value {
                    if i < num_train_rows {
                        &mut user_train
                    } else {
                        &mut user_test
                    }
                } else if column.name == assistant_column_value {
                    if i < num_train_rows {
                        &mut assistant_train
                    } else {
                        &mut assistant_test
                    }
                } else {
                    continue;
                };

                match column.duckdb_type.as_str() {
                    "bpchar" | "text" | "varchar" => {
                        match row.get::<_, Option<String>>(column.position).unwrap() {
                            Some(text) => vector.push(text),
                            None => panic!("NULL training text is not handled"),
                        }
                    }
                    _ => panic!("only text type columns are supported"),
                }
            }
            i += 1;
        }

        ConversationDataset {
            system_train,
            user_train,
            assistant_train,
            system_test,
            user_test,
            assistant_test,
            num_features,
            num_rows,
            num_test_rows,
            num_train_rows,
        }
    }

    pub fn tabular_dataset(&mut self) -> Dataset {
        let numeric_encoded_dataset = self.numeric_encoded_dataset();

        let label_data = ndarray::ArrayView2::from_shape(
            (
                numeric_encoded_dataset.num_train_rows,
                numeric_encoded_dataset.num_labels,
            ),
            &numeric_encoded_dataset.y_train,
        )
        .unwrap();

        let feature_data = ndarray::ArrayView2::from_shape(
            (
                numeric_encoded_dataset.num_train_rows,
                numeric_encoded_dataset.num_features,
            ),
            &numeric_encoded_dataset.x_train,
        )
        .unwrap();

        // We only analyze supervised training sets that have labels for now.
        if numeric_encoded_dataset.num_labels > 0 {
            // We only analyze features against the first label in joint regression.
            let target_data = label_data.columns().into_iter().next().unwrap();

            // Analyze labels
            Zip::from(label_data.columns())
                .and(&self.label_positions())
                .for_each(|data, position| {
                    let column = &mut self.columns[position.column_position - 1]; // lookup the mutable one
                    column.analyze(&data, &target_data);
                });

            // Analyze features
            Zip::from(feature_data.columns())
                .and(&self.feature_positions())
                .for_each(|data, position| {
                    let column = &mut self.columns[position.column_position - 1]; // lookup the mutable one
                    column.analyze(&data, &target_data);
                });
        } else {
            // Analyze features for unsupervised learning
            Zip::from(feature_data.columns())
                .and(&self.feature_positions())
                .for_each(|data, position| {
                    let column = &mut self.columns[position.column_position - 1]; // lookup the mutable one
                    column.analyze(&data, &data);
                });
        }

        let mut analysis = IndexMap::new();
        analysis.insert(
            "samples".to_string(),
            numeric_encoded_dataset.num_rows as f32,
        );
        self.analysis = Some(analysis);

        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
        let analysis_param = serde_json::to_string(&self.analysis).unwrap();
        let columns_param = serde_json::to_string(&self.columns).unwrap();
        conn.execute(
            "UPDATE quackml.snapshots SET analysis = $1, columns = $2 WHERE id = $3",
            params![analysis_param, columns_param, &self.id,],
        )
        .unwrap();
        // Record the analysis

        let features_width = self.features_width();
        let mut x_train = vec![0_f32; features_width * numeric_encoded_dataset.num_train_rows];
        Zip::from(feature_data.columns())
            .and(&self.feature_positions())
            .for_each(|data, position| {
                let column = &self.columns[position.column_position - 1];
                column.preprocess(&data, &mut x_train, features_width, position.row_position);
            });

        let mut x_test = vec![0_f32; features_width * numeric_encoded_dataset.num_test_rows];
        let test_features = ndarray::ArrayView2::from_shape(
            (
                numeric_encoded_dataset.num_test_rows,
                numeric_encoded_dataset.num_features,
            ),
            &numeric_encoded_dataset.x_test,
        )
        .unwrap();
        Zip::from(test_features.columns())
            .and(&self.feature_positions())
            .for_each(|data, position| {
                let column = &self.columns[position.column_position - 1];
                column.preprocess(&data, &mut x_test, features_width, position.row_position);
            });

        self.feature_positions = self.feature_positions();

        Dataset {
            x_train,
            x_test,
            num_distinct_labels: self.num_classes(), // changes after analysis
            ..numeric_encoded_dataset
        }
    }

    // Encodes categorical text values (and all others) into f32 for memory efficiency and type homogenization.
    pub fn numeric_encoded_dataset(&mut self) -> Dataset {
        let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
        let mut stmt = conn.prepare(&self.select_sql()).unwrap();
        let num_rows = stmt.query_and_then([], |row| -> Result<_, _> {
            let val= row.get::<_, duckdb::types::Value>(0);
            val
        }).unwrap().count();
        let (num_train_rows, num_test_rows) = self.train_test_split(num_rows);
        let num_features = self.num_features();
        let num_labels = self.num_labels();

        let mut x_train: Vec<f32> = Vec::with_capacity(num_train_rows * num_features);
        let mut y_train: Vec<f32> = Vec::with_capacity(num_train_rows * num_labels);
        let mut x_test: Vec<f32> = Vec::with_capacity(num_test_rows * num_features);
        let mut y_test: Vec<f32> = Vec::with_capacity(num_test_rows * num_labels);

        let mut rows = stmt.query([]).unwrap();
        let mut i = 0;
        while let Some(row) = rows.next().unwrap() {
            let row = row;

            for column in &mut self.columns {
                let vector = if column.label {
                    if i < num_train_rows {
                        &mut y_train
                    } else {
                        &mut y_test
                    }
                } else if i < num_train_rows {
                    &mut x_train
                } else {
                    &mut x_test
                };

                match &mut column.statistics.categories {
                    // Categorical encoding types
                    Some(categories) => {
                        let key = match column.duckdb_type.as_str() {
                            "BOOLEAN" => row.get::<_, bool>(column.position).map(|v| v.to_string()),
                            "SMALLINT" => row.get::<_, i16>(column.position).map(|v| v.to_string()),
                            "INTEGER" => row.get::<_, i32>(column.position).map(|v| v.to_string()),
                            "BIGINT" => row.get::<_, i64>(column.position).map(|v| v.to_string()),
                            "REAL" => row.get::<_, f32>(column.position).map(|v| v.to_string()),
                            "DOUBLE" => row.get::<_, f64>(column.position).map(|v| v.to_string()),
                            "VARCHAR" | "CHAR" => {
                                row.get::<_, String>(column.position).map(|v| v.to_string())
                            }
                            _ => panic!(
                                "Unhandled type for categorical variable: {} {:?}",
                                column.name, column.duckdb_type
                            ),
                            _ => panic!(
                                "Unhandled type for categorical variable: {} {:?}",
                                column.name, column.duckdb_type
                            ),
                        };
                        let key = key.unwrap_or_else(|_| NULL_CATEGORY_KEY.to_string());
                        if i < num_train_rows {
                            let len = categories.len();
                            let category = categories.entry(key).or_insert_with_key(|key| {
                                let value = match key.as_str() {
                                    NULL_CATEGORY_KEY => 0_f32, // NULL values are always Category 0
                                    _ => match &column.preprocessor.encode {
                                        Encode::target | Encode::native | Encode::one_hot { .. } => len as f32,
                                        Encode::ordinal(values) => {
                                            match values.iter().position(|v| v == key.as_str()) {
                                                Some(i) => (i + 1) as f32,
                                                None => panic!(
                                                    "value is not present in ordinal: {:?}. Valid values: {:?}",
                                                    key, values
                                                ),
                                            }
                                        }
                                    },
                                };
                                Category { value, members: 0 }
                            });
                            category.members += 1;
                            vector.push(category.value);
                        } else {
                            vector.push(column.get_category_value(&key));
                        }
                    }

                    // All quantitative and native types are cast directly to f32
                    None => {
                        if column.array {
                            match column.duckdb_type.as_str() {
                                // TODO handle NULL in arrays
                                "BOOLEAN[]" => {
                                    let vec = row.get::<_, Vec<_>>(column.position).unwrap();

                                    check_column_size(column, vec.len());
                                    for j in vec {
                                        vector.push(j as u8 as f32)
                                    }
                                }
                                "SMALLINT[]" => {
                                    let vec = row.get::<_, Vec<_>>(column.position).unwrap();

                                    check_column_size(column, vec.len());

                                    for j in vec {
                                        vector.push(j as f32)
                                    }
                                }
                                "INTEGER[]" => {
                                    let vec = row.get::<_, Vec<_>>(column.position).unwrap();

                                    check_column_size(column, vec.len());

                                    for j in vec {
                                        vector.push(j as f32)
                                    }
                                }
                                "BIGINT[]" => {
                                    let vec = row.get::<_, Vec<_>>(column.position).unwrap();

                                    check_column_size(column, vec.len());

                                    for j in vec {
                                        vector.push(j as f32)
                                    }
                                }
                                "REAL[]" => {
                                    let vec = row.get::<_, Vec<_>>(column.position).unwrap();

                                    check_column_size(column, vec.len());

                                    for j in vec {
                                        vector.push(j.into())
                                    }
                                }
                                "DOUBLE[]" => {
                                    let vec = row.get::<_, Vec<_>>(column.position).unwrap();

                                    check_column_size(column, vec.len());

                                    for j in vec {
                                        vector.push(j as f32)
                                    }
                                }
                                _ => panic!(
                                    "Unhandled type for quantitative array column: {} {:?}",
                                    column.name, column.duckdb_type
                                ),
                            }
                        } else {
                            // scalar
                            let float = match column.duckdb_type.as_str() {
                                "BOOLEAN" => row.get(column.position).unwrap(),
                                "SMALLINT" => row.get(column.position).unwrap(),
                                "INTEGER" => row.get(column.position).unwrap(),
                                "BIGINT" => row.get(column.position).unwrap(),
                                "REAL" => row.get(column.position).unwrap(),
                                "DOUBLE" => row.get(column.position).unwrap(),
                                "DECIMAL" => row.get(column.position).unwrap(),

                                _ => panic!(
                                    "Unhandled type for quantitative scalar column: {} {:?}",
                                    column.name, column.duckdb_type
                                ),
                            };
                            match float {
                                Some(f) => vector.push(f),
                                None => vector.push(f32::NAN),
                            }
                        }
                    }
                }
            }

            i += 1;
        }
        // recompute the number of features now that we know array widths
        let num_features = self.num_features();
        let num_labels = self.num_labels();

        Dataset {
            x_train,
            y_train,
            x_test,
            y_test,
            num_features,
            num_labels,
            num_rows,
            num_test_rows,
            num_train_rows,
            // TODO rename and audit this
            num_distinct_labels: self.num_classes(),
        }
    }

    pub fn snapshot_name(&self) -> String {
        format!("\"pgml\".\"snapshot_{}\"", self.id)
    }

    pub fn relation_name(&self) -> String {
        match self.materialized {
            true => self.snapshot_name(),
            false => self.relation_name.clone(),
        }
    }

    fn relation_name_quoted(&self) -> String {
        match self.materialized {
            true => self.snapshot_name(), // Snapshot name is already safe.
            false => {
                let (schema_name, table_name) = Self::fully_qualified_table(&self.relation_name);
                format!("\"{}\".\"{}\"", schema_name, table_name)
            }
        }
    }
}

#[inline]
fn check_column_size(column: &mut Column, len: usize) {
    if column.size == 0 {
        column.size = len;
    } else if column.size != len {
        panic!(
            "Mismatched array length for feature `{}`. Expected: {} Received: {}",
            column.name, column.size, len
        );
    }
}
