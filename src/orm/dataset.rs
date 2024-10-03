use crate::context::{self,  DATABASE_CONTEXT};
use core::panic;
use duckdb::params;
use flate2::read::GzDecoder;
use lazy_static::lazy_static;
use serde::Deserialize;
use std::collections::HashMap;
use std::env::temp_dir;
use std::fmt::{Display, Formatter};
use std::io::{Read, Write};
use std::path;

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
        conn.execute_batch(&format!("DROP TABLE IF EXISTS quackml.{}", table_name))
            .unwrap();
    }
}

lazy_static! {
    static ref DATASETS: HashMap<&'static str, &'static [u8]> = {
        let mut map: HashMap<&str, &[u8]> = HashMap::new();
        map.insert(
            "breast_cancer.csv",
            include_bytes!("datasets/breast_cancer.csv.gz"),
        );
        map.insert("diabetes.csv", include_bytes!("datasets/diabetes.csv.gz"));
        map.insert("digits.csv", include_bytes!("datasets/digits.csv.gz"));
        map.insert("iris.csv", include_bytes!("datasets/iris.csv.gz"));
        map.insert("linnerud.csv", include_bytes!("datasets/linnerud.csv.gz"));
        map.insert("wine.csv", include_bytes!("datasets/wine.csv.gz"));
        map
    };
}
#[derive(Deserialize)]
struct BreastCancerRow {
    mean_radius: f32,
    mean_texture: f32,
    mean_perimeter: f32,
    mean_area: f32,
    mean_smoothness: f32,
    mean_compactness: f32,
    mean_concavity: f32,
    mean_concave_points: f32,
    mean_symmetry: f32,
    mean_fractal_dimension: f32,
    radius_error: f32,
    texture_error: f32,
    perimeter_error: f32,
    area_error: f32,
    smoothness_error: f32,
    compactness_error: f32,
    concavity_error: f32,
    concave_points_error: f32,
    symmetry_error: f32,
    fractal_dimension_error: f32,
    worst_radius: f32,
    worst_texture: f32,
    worst_perimeter: f32,
    worst_area: f32,
    worst_smoothness: f32,
    worst_compactness: f32,
    worst_concavity: f32,
    worst_concave_points: f32,
    worst_symmetry: f32,
    worst_fractal_dimension: f32,
    target: i32,
}

pub fn load_breast_cancer(limit: Option<usize>) -> (String, i64) {
    drop_table_if_exists("breast_cancer");
    context::run(|conn| {
        conn.execute(
            r#"CREATE TABLE quackml.breast_cancer (
        "mean radius" FLOAT, 
        "mean texture" FLOAT, 
        "mean perimeter" FLOAT, 
        "mean area" FLOAT,
        "mean smoothness" FLOAT,
        "mean compactness" FLOAT,
        "mean concavity" FLOAT,
        "mean concave points" FLOAT,
        "mean symmetry" FLOAT,
        "mean fractal dimension" FLOAT,
        "radius error" FLOAT,
        "texture error" FLOAT,
        "perimeter error" FLOAT,
        "area error" FLOAT,
        "smoothness error" FLOAT,
        "compactness error" FLOAT,
        "concavity error" FLOAT,
        "concave points error" FLOAT,
        "symmetry error" FLOAT,
        "fractal dimension error" FLOAT,
        "worst radius" FLOAT,
        "worst texture" FLOAT,
        "worst perimeter" FLOAT,
        "worst area" FLOAT,
        "worst smoothness" FLOAT,
        "worst compactness" FLOAT,
        "worst concavity" FLOAT,
        "worst concave points" FLOAT,
        "worst symmetry" FLOAT,
        "worst fractal dimension" FLOAT,
        "malignant" BOOLEAN
    )"#,
            params![],
        )
        .map_err(|e| anyhow::anyhow!(e))
    })
    .unwrap();

    let limit = match limit {
        Some(limit) => limit,
        None => usize::MAX,
    };

    let data: &[u8] = std::include_bytes!("datasets/breast_cancer.csv.gz");
    let decoder = GzDecoder::new(data);
    let mut csv = csv::ReaderBuilder::new().from_reader(decoder);

    let mut inserted = 0;
    for (i, row) in csv.deserialize().enumerate() {
        if i >= limit {
            break;
        }
        let row: BreastCancerRow = row.unwrap();
        context::run(|conn| {
        conn.execute(
            r#"
        INSERT INTO quackml.breast_cancer ("mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness", "mean compactness", "mean concavity", "mean concave points", "mean symmetry", 
            "mean fractal dimension", "radius error", "texture error", "perimeter error", "area error", "smoothness error", "compactness error", "concavity error", "concave points error", "symmetry error", 
            "fractal dimension error", "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness", "worst compactness", "worst concavity", "worst concave points", "worst symmetry", 
            "worst fractal dimension", "malignant") 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
            params![
                row.mean_radius, row.mean_texture, row.mean_perimeter, row.mean_area,
                row.mean_smoothness, row.mean_compactness, row.mean_concavity, row.mean_concave_points,
                row.mean_symmetry, row.mean_fractal_dimension, row.radius_error, row.texture_error,
                row.perimeter_error, row.area_error, row.smoothness_error, row.compactness_error,
                row.concavity_error, row.concave_points_error, row.symmetry_error,
                row.fractal_dimension_error, row.worst_radius, row.worst_texture, row.worst_perimeter,
                row.worst_area, row.worst_smoothness, row.worst_compactness, row.worst_concavity,
                row.worst_concave_points, row.worst_symmetry, row.worst_fractal_dimension,
                row.target == 0
            ]).map_err(|e| anyhow::anyhow!(e))
        }
        ).unwrap();
        inserted += 1;
    }

    ("quackml.breast_cancer".to_string(), inserted)
}

#[derive(Deserialize)]
struct DiabetesRow {
    age: f32,
    sex: f32,
    bmi: f32,
    bp: f32,
    s1: f32,
    s2: f32,
    s3: f32,
    s4: f32,
    s5: f32,
    s6: f32,
    target: f32,
}

pub fn load_diabetes(limit: Option<usize>) -> (String, i64) {
    drop_table_if_exists("diabetes");
    context::run(|conn| {
        conn.execute(
            "CREATE TABLE pgml.diabetes (
        age FLOAT4, 
        sex FLOAT4, 
        bmi FLOAT4, 
        bp FLOAT4, 
        s1 FLOAT4, 
        s2 FLOAT4, 
        s3 FLOAT4, 
        s4 FLOAT4, 
        s5 FLOAT4, 
        s6 FLOAT4, 
        target FLOAT4
    )",
            params![],
        )
        .map_err(|e| anyhow::anyhow!(e))
    })
    .unwrap();

    let limit = match limit {
        Some(limit) => limit,
        None => usize::MAX,
    };

    let data: &[u8] = std::include_bytes!("datasets/diabetes.csv.gz");
    let decoder = GzDecoder::new(data);
    let mut csv = csv::ReaderBuilder::new().from_reader(decoder);

    let mut inserted = 0;
    for (i, row) in csv.deserialize().enumerate() {
        if i >= limit {
            break;
        }
        let row: DiabetesRow = row.unwrap();
        context::run(|conn| {
            conn.execute(
                r#"
            INSERT INTO quackml.diabetes (age, sex, bmi, bp, s1, s2, s3, s4, s5, s6, target) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            "#,
                params![
                    row.age, row.sex, row.bmi, row.bp, row.s1, row.s2, row.s3, row.s4, row.s5,
                    row.s6, row.target
                ],
            )
            .map_err(|e| anyhow::anyhow!(e))
        })
        .unwrap();
        inserted += 1;
    }

    ("quackml.diabetes".to_string(), inserted)
}

#[derive(Deserialize)]
struct DigitsRow {
    image: String,
    target: i16,
}

pub fn load_digits(limit: Option<usize>) -> (String, i64) {
    drop_table_if_exists("digits");
    context::run(|conn| {
        conn.execute(
            "CREATE TABLE quackml.digits (image INTEGER[][], target SMALLINT)",
            params![],
        )
        .map_err(|e| anyhow::anyhow!(e))
    })
    .unwrap();

    let limit = match limit {
        Some(limit) => limit,
        None => usize::MAX,
    };

    let data: &[u8] = std::include_bytes!("datasets/digits.csv.gz");
    let decoder = GzDecoder::new(data);
    let mut csv = csv::ReaderBuilder::new().from_reader(decoder);

    let mut inserted = 0;
    for (i, row) in csv.deserialize().enumerate() {
        if i >= limit {
            break;
        }
        let row: DigitsRow = row.unwrap();
        context::run(|conn| {
            conn.execute(
                "
            INSERT INTO quackml.digits (image, target)
            VALUES (?, ?)
            ",
                params![row.image, row.target],
            )
            .map_err(|e| anyhow::anyhow!(e))
        })
        .unwrap();
        inserted += 1;
    }

    ("quackml.digits".to_string(), inserted)
}

#[derive(Deserialize)]
struct IrisRow {
    sepal_length: f32,
    sepal_width: f32,
    petal_length: f32,
    petal_width: f32,
    target: i32,
}

pub fn load_iris(limit: Option<usize>) -> (String, i64) {
    drop_table_if_exists("iris");
    context::run(|conn| {
        conn.execute(
            "CREATE TABLE quackml.iris (
        sepal_length FLOAT, 
        sepal_width FLOAT, 
        petal_length FLOAT, 
        petal_width FLOAT, 
        target INTEGER
    )",
            params![],
        )
        .map_err(|e| anyhow::anyhow!(e))
    })
    .unwrap();

    let limit = match limit {
        Some(limit) => limit,
        None => usize::MAX,
    };

    let data: &[u8] = std::include_bytes!("datasets/iris.csv.gz");
    let decoder = GzDecoder::new(data);
    let mut csv = csv::ReaderBuilder::new().from_reader(decoder);

    let mut inserted = 0;
    for (i, row) in csv.deserialize().enumerate() {
        if i >= limit {
            break;
        }
        let row: IrisRow = row.unwrap();
        context::run(|conn| {
            conn.execute(
                "
        INSERT INTO quackml.iris (sepal_length, sepal_width, petal_length, petal_width, target)
        VALUES (?, ?, ?, ?, ?)
        ",
                params![
                    row.sepal_length,
                    row.sepal_width,
                    row.petal_length,
                    row.petal_width,
                    row.target
                ],
            )
            .map_err(|e| anyhow::anyhow!(e))
        })
        .unwrap();
        inserted += 1;
    }

    ("quackml.iris".to_string(), inserted)
}

#[derive(Deserialize)]
struct LinnerudRow {
    chins: f32,
    situps: f32,
    jumps: f32,
    weight: f32,
    waist: f32,
    pulse: f32,
}

pub fn load_linnerud(limit: Option<usize>) -> (String, i64) {
    drop_table_if_exists("linnerud");
    context::run(|conn| {
        conn.execute(
            "CREATE TABLE quackml.linnerud(
        chins FLOAT,
        situps FLOAT,
        jumps FLOAT,
        weight FLOAT,
        waist FLOAT,
        pulse FLOAT
    )",
            params![],
        )
        .map_err(|e| anyhow::anyhow!(e))
    })
    .unwrap();

    let limit = match limit {
        Some(limit) => limit,
        None => usize::MAX,
    };

    let data: &[u8] = std::include_bytes!("datasets/linnerud.csv.gz");
    let decoder = GzDecoder::new(data);
    let mut csv = csv::ReaderBuilder::new().from_reader(decoder);

    let mut inserted = 0;
    for (i, row) in csv.deserialize().enumerate() {
        if i >= limit {
            break;
        }
        let row: LinnerudRow = row.unwrap();
        context::run(|conn| {
            conn.execute(
                "
        INSERT INTO quackml.linnerud (chins, situps, jumps, weight, waist, pulse)
        VALUES (?, ?, ?, ?, ?, ?)
        ",
                params![row.chins, row.situps, row.jumps, row.weight, row.waist, row.pulse],
            )
            .map_err(|e| anyhow::anyhow!(e))
        })
        .unwrap();
        inserted += 1;
    }

    ("quackml.linnerud".to_string(), inserted)
}

#[derive(Deserialize)]
struct WineRow {
    alcohol: f32,
    malic_acid: f32,
    ash: f32,
    alcalinity_of_ash: f32,
    magnesium: f32,
    total_phenols: f32,
    flavanoids: f32,
    nonflavanoid_phenols: f32,
    proanthocyanins: f32,
    hue: f32,
    color_intensity: f32,
    od280_od315_of_diluted_wines: f32,
    proline: f32,
    target: i32,
}

pub fn load_wine(limit: Option<usize>) -> (String, i64) {
    drop_table_if_exists("wine");
    context::run(|conn| {
        conn.execute(
            r#"CREATE TABLE quackml.wine (
            alcohol FLOAT,
            malic_acid FLOAT,
            ash FLOAT,
            alcalinity_of_ash FLOAT,
            magnesium FLOAT,
            total_phenols FLOAT,
            flavanoids FLOAT,
            nonflavanoid_phenols FLOAT,
            proanthocyanins FLOAT,
            hue FLOAT,
            color_intensity FLOAT,
            "od280/od315_of_diluted_wines" FLOAT,
            proline FLOAT,
            target INTEGER
        )"#,
            params![],
        )
        .map_err(|e| anyhow::anyhow!(e))
    })
    .unwrap();

    let limit = match limit {
        Some(limit) => limit,
        None => usize::MAX,
    };

    let data: &[u8] = std::include_bytes!("datasets/wine.csv.gz");
    let decoder = GzDecoder::new(data);
    let mut csv = csv::ReaderBuilder::new().from_reader(decoder);

    let mut inserted = 0;
    for (i, row) in csv.deserialize().enumerate() {
        if i >= limit {
            break;
        }
        let row: WineRow = row.unwrap();
        context::run(|conn| conn
            .execute(
            r#"
        INSERT INTO quackml.wine (alcohol, malic_acid, ash, alcalinity_of_ash, magnesium, total_phenols, flavanoids, nonflavanoid_phenols, proanthocyanins, color_intensity, hue, "od280/od315_of_diluted_wines", proline, target) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        "#,
            params![
                row.alcohol, row.malic_acid, row.ash, row.alcalinity_of_ash, row.magnesium,
                row.total_phenols, row.flavanoids, row.nonflavanoid_phenols, row.proanthocyanins,
                row.color_intensity, row.hue, row.od280_od315_of_diluted_wines, row.proline, row.target
            ]).map_err(|e| anyhow::anyhow!(e))
                    
        ).unwrap();
        inserted += 1;
    }

    ("quackml.wine".to_string(), inserted)
}

pub fn load_datasets() {
    let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
    DATASETS.iter().for_each(|(name, data)| {
        let mut decoder = GzDecoder::new(&data[..]);
        let mut csv = String::new();
        decoder.read_to_string(&mut csv).unwrap();
        println!("Read {} bytes from {}", csv.len(), name);
        let tmp_dir = temp_dir();
        let file_path = tmp_dir.join(*name);
        let mut file = std::fs::File::create(&file_path).unwrap();
        file.write_all(csv.as_bytes()).unwrap();
        let table_name = name.replace(".csv", "");
        drop_table_if_exists(&table_name);
        conn.execute_batch(
            format!(
                "CREATE TABLE quackml.{} AS SELECT * FROM read_csv('{}')",
                table_name,
                file_path.to_str().unwrap()
            )
            .as_str(),
        )
        .unwrap();
        std::fs::remove_file(file_path).unwrap(); // Clean up temporary file
    });
}
