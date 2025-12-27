# CLAUDE.md - AI Assistant Guide for quackML

## Project Overview

**quackML** is a DuckDB extension implementing a full-service AI/ML engine written in Rust. It enables developing AI/ML models in pure SQL with DuckDB, bringing models to data rather than vice versa.

### Supported Capabilities
- **Traditional ML**: Linear/logistic regression, XGBoost, LightGBM, SVM, random forests, clustering, decomposition
- **Deep Learning**: HuggingFace Transformers integration for text tasks
- **NLP Tasks**: Text classification, text generation, summarization, translation, question answering, embeddings
- **Fine-tuning**: Support for fine-tuning models like GPT-2 on custom data

## Codebase Architecture

```
quackML/
├── src/
│   ├── lib.rs              # Extension entrypoint, registers SQL functions
│   ├── api.rs              # SQL function implementations (train, finetune, predict, etc.)
│   ├── bindings/           # ML library bindings
│   │   ├── mod.rs          # Bindings trait definition
│   │   ├── linfa.rs        # Pure Rust ML (linear/logistic regression, SVM)
│   │   ├── xgboost.rs      # XGBoost bindings
│   │   ├── lightgbm.rs     # LightGBM bindings
│   │   ├── sklearn/        # Scikit-learn Python bindings
│   │   ├── transformers/   # HuggingFace Transformers bindings
│   │   ├── python/         # Python/venv management
│   │   └── langchain/      # LangChain integration
│   ├── context/            # Database connection context management
│   │   └── context.rs      # Global DATABASE_CONTEXT for connection access
│   ├── orm/                # Data models and database operations
│   │   ├── mod.rs          # Re-exports all ORM types
│   │   ├── project.rs      # Project management (ML projects)
│   │   ├── model.rs        # Model storage and deployment
│   │   ├── snapshot.rs     # Data snapshots for training
│   │   ├── dataset.rs      # Dataset loading and management
│   │   ├── algorithm.rs    # Supported algorithm enum
│   │   ├── task.rs         # ML task types (regression, classification, etc.)
│   │   ├── metrics.rs      # Model evaluation metrics
│   │   ├── sampling.rs     # Train/test sampling strategies
│   │   ├── search.rs       # Hyperparameter search
│   │   ├── strategy.rs     # Deployment strategies
│   │   └── status.rs       # Job status tracking
│   └── sql/
│       └── schema.sql      # Database schema (tables, views, enums)
├── duckdb-rs/              # Git submodule: forked duckdb-rs with extension support
├── Cargo.toml              # Rust dependencies and workspace config
├── requirements.txt        # Python dependencies
└── *.csv                   # Sample datasets (iris, digits, diabetes, etc.)
```

## Key Modules Explained

### `src/lib.rs` - Extension Entry Point
- Defines `quack_ml_init()` as the DuckDB extension entrypoint
- Initializes database context and schema
- Registers all SQL functions: `train`, `finetune`, `predict`, `predict_text`, `predict_proba`, `embed`, `transform`, `generate`, `load_dataset`

### `src/api.rs` - SQL Function Implementations
- Implements DuckDB virtual table functions (`VTab`) for `train` and `finetune`
- Implements scalar functions for predictions and embeddings
- Handles parameter binding and result generation
- Large file (~35k tokens) - read in chunks if needed

### `src/bindings/` - ML Library Integrations

**Core Trait** (`mod.rs`):
```rust
pub trait Bindings: Send + Sync + Debug {
    fn predict(&self, features: &[f32], num_features: usize, num_classes: usize) -> Result<Vec<f32>>;
    fn predict_proba(&self, features: &[f32], num_features: usize) -> Result<Vec<f32>>;
    fn to_bytes(&self) -> Result<Vec<u8>>;
    fn from_bytes(bytes: &[u8]) -> Result<Box<dyn Bindings>>;
}
```

**Implementations**:
- `linfa.rs`: Pure Rust implementations (LinearRegression, LogisticRegression, Svm)
- `xgboost.rs`: XGBoost gradient boosting
- `lightgbm.rs`: LightGBM gradient boosting
- `sklearn/mod.rs`: Scikit-learn via PyO3 (many algorithms)
- `transformers/mod.rs`: HuggingFace models via PyO3

### `src/orm/` - Data Models

**Key Types**:
- `Project`: ML project container (name, task type)
- `Model`: Trained model with hyperparams, metrics, serialized bindings
- `Snapshot`: Frozen dataset for reproducible training
- `Dataset`: In-memory training/test data splits
- `Algorithm`: Enum of all supported algorithms (49 variants)
- `Task`: ML task types (regression, classification, text_classification, etc.)

### `src/context/context.rs` - Database Context
- Global `DATABASE_CONTEXT` for connection access
- `context::run(|conn| ...)` pattern for database operations

## SQL Schema (`src/sql/schema.sql`)

**Tables**:
- `quackml.projects`: ML project definitions
- `quackml.models`: Trained models with metrics and hyperparams
- `quackml.snapshots`: Data snapshots for training
- `quackml.deployments`: Model deployment history
- `quackml.logs`: Training logs
- `quackml.files`: Serialized model file storage

**Custom Types**:
- `task`: regression, classification, text_classification, embedding, etc.
- `sampling`: random, stratified, time_series, last
- `strategy`: new_score, best_score, most_recent, rollback, specific
- `status`: pending, in_progress, running, successful, failed

**Views**:
- `quackml.overview`: Current deployment status
- `quackml.trained_models`: All trained models with details
- `quackml.deployed_models`: Currently deployed models

## Development Workflow

### Building
```bash
# Initialize submodules (required for duckdb-rs)
git submodule update --init --recursive

# Build the extension
cargo build --release

# The extension will be at: target/release/libquack_ml.so (Linux)
```

### Python Environment
The extension requires a Python virtual environment with ML dependencies:
```bash
python3 -m venv quackml-venv
source quackml-venv/bin/activate
pip install -r requirements.txt
```

### Loading the Extension in DuckDB
```sql
LOAD 'path/to/libquack_ml.so';
```

## Code Conventions

### Rust Style
- Edition 2021
- Snake_case for enum variants (e.g., `Algorithm::linear`, `Task::text_classification`)
- Use `anyhow::Result` for error handling
- PyO3 for Python interop with `#[cfg(feature = "python")]` gates

### Python Bindings Pattern
```rust
// Create Python module from embedded source
create_pymodule!("/src/bindings/sklearn/sklearn.py");

// Call Python function
Python::with_gil(|py| {
    let module = get_module!(PY_MODULE);
    let result = module.getattr(py, "function_name")?.call1(py, args)?;
    result.extract(py)
})
```

### Database Access Pattern
```rust
// Use context::run for database operations
context::run(|conn| {
    conn.query_row("SELECT ...", params![...], |row| Ok(row.get(0)?))
})

// Or access directly via DATABASE_CONTEXT
let conn = unsafe { DATABASE_CONTEXT.as_ref().unwrap().get_connection() };
```

### ORM Patterns
- Each ORM type has `create()`, `find()`, `find_by_*()` methods
- Models are cached in memory via `Lazy<Mutex<HashMap>>`
- Serialization uses `rmp_serde` (MessagePack) for Rust types, pickle for Python

## Key Dependencies

### Rust
- `duckdb` / `libduckdb-sys`: DuckDB integration (forked for extension support)
- `pyo3`: Python interop
- `linfa` / `linfa-*`: Pure Rust ML algorithms
- `xgboost`: XGBoost bindings
- `lightgbm`: LightGBM bindings
- `ndarray`: N-dimensional arrays
- `serde` / `serde_json`: Serialization

### Python (via PyO3)
- `torch`, `transformers`: Deep learning
- `scikit-learn`: Traditional ML
- `xgboost`, `lightgbm`, `catboost`: Gradient boosting
- `sentence-transformers`: Embeddings
- `datasets`: HuggingFace datasets

## Git Submodules

The project uses a forked `duckdb-rs` as a git submodule:
- Path: `duckdb-rs/`
- Branch: `poc-rust-c-extension-api`
- Contains DuckDB Rust bindings with loadable extension support

Always run `git submodule update --init --recursive` after cloning.

## SQL API Examples

### Training a Model
```sql
SELECT * FROM train(
    'my_project',
    task => 'classification',
    relation_name => 'my_table',
    y_column_name => 'target',
    algorithm => 'xgboost'
);
```

### Making Predictions
```sql
SELECT predict('my_project', feature1, feature2, ...) FROM my_data;
```

### Fine-tuning Transformers
```sql
SELECT * FROM finetune(
    'IMDB Review Sentiment',
    task => 'text_classification',
    relation_name => 'quackml.glue_data',
    y_column_name => 'class',
    model_name => 'gpt2',
    hyperparams => '{"training_args": {"learning_rate": 2e-5}}'
);
```

### Loading HuggingFace Datasets
```sql
SELECT load_dataset('imdb');
```

### Generating Text
```sql
SELECT generate(model_id, 'prompt text');
```

### Creating Embeddings
```sql
SELECT embed('sentence-transformers/all-MiniLM-L6-v2', text_column) FROM my_table;
```

## Important Notes for AI Assistants

1. **Large Files**: `src/api.rs` is very large (~35k tokens). Read in chunks using `offset` and `limit` parameters.

2. **Python Feature Flag**: Most Python-dependent code is behind `#[cfg(feature = "python")]`. The `python` feature is enabled by default.

3. **Unsafe Code**: Database context uses `unsafe` for global state access - this is intentional for the extension architecture.

4. **Submodule Required**: The `duckdb-rs` submodule must be initialized for the project to compile.

5. **Schema Changes**: When modifying the SQL schema, update `src/sql/schema.sql` - it's loaded at extension init via `include_str!`.

6. **Algorithm Support**: Adding new algorithms requires:
   - Adding variant to `Algorithm` enum in `src/orm/algorithm.rs`
   - Implementing `Bindings` trait
   - Adding fit function to appropriate bindings module
   - Updating `Model::fit()` match statement in `src/orm/model.rs`

7. **Python Bindings**: Python code is embedded as strings via `include_str!` and executed at runtime. Source files are in the same directory as their Rust modules.

8. **Testing**: The project includes sample CSV datasets for testing (iris, diabetes, digits, etc.).

## Project Status

The project is still in development. Core functionality for training and inference is implemented, but some features may be incomplete or experimental.
