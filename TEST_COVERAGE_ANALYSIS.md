# Test Coverage Analysis for quackML

## Executive Summary

The quackML codebase has **extremely limited test coverage** with only **2 active test functions** across approximately **8,500+ lines of Rust code**. This represents critical risk for a machine learning extension where algorithm correctness and data handling are paramount.

---

## Current Test Status

### Active Tests (2 total)

| File | Test | What It Tests |
|------|------|---------------|
| `src/context/context.rs` | `test_database_context()` | Database context initialization and connection cloning |
| `src/orm/metrics.rs` | `test_confusion_matrix_multiclass()` | F1 score calculation with perfect predictions (4 classes) |

### Commented-Out Tests

- **`src/api.rs`**: ~20 test functions commented out - these were PostgreSQL-specific (`#[pg_test]`) and don't compile with DuckDB
- **`src/bindings/transformers/whitelist.rs`**: 3 test functions commented out

### Test Infrastructure

| Component | Status |
|-----------|--------|
| `[dev-dependencies]` in Cargo.toml | **Missing** |
| Test utilities/helpers | **None** |
| Test fixtures/factories | **None** |
| Integration tests directory (`tests/`) | **None** |
| Mocking framework | **None** |
| CI/CD test pipeline | **None** |

---

## Critical Gaps by Priority

### Priority 1: Core ML Functionality (Highest Risk)

#### 1.1 Algorithm Bindings - 0% Coverage

These implement the core ML value proposition and have **zero tests**:

| Module | Lines | Algorithms |
|--------|-------|------------|
| `src/bindings/linfa.rs` | 328 | LinearRegression, LogisticRegression, SVM |
| `src/bindings/xgboost.rs` | 370 | XGBoost gradient boosting |
| `src/bindings/lightgbm.rs` | 146 | LightGBM gradient boosting |
| `src/bindings/sklearn/mod.rs` | 379 | 40+ scikit-learn algorithms |
| `src/bindings/transformers/mod.rs` | 705 | HuggingFace transformers |

**Recommended tests:**
- Fit/predict round-trip for each algorithm with known datasets
- Serialization/deserialization of trained models
- Edge cases: empty data, single sample, mismatched dimensions
- Numeric precision validation against reference implementations

#### 1.2 Model Management (`src/orm/model.rs`) - 0% Coverage

The `Model` struct (1,002 lines) handles training orchestration, prediction, and deployment:

**Recommended tests:**
- `Model::fit()` for each algorithm type
- `Model::predict()` output shape and value ranges
- `Model::predict_proba()` probability sum validation
- Model caching (`MODELS` HashMap) behavior
- Deployment logic (`deploy()`, `find_deployed()`)
- Hyperparameter parsing and validation

#### 1.3 Data Snapshots (`src/orm/snapshot.rs`) - 0% Coverage

Snapshot management (1,515 lines) handles data preprocessing:

**Recommended tests:**
- Feature extraction from different column types
- Train/test splitting with different sampling strategies
- Null/missing value handling
- Data type conversions (categorical encoding, normalization)
- Snapshot caching and retrieval

### Priority 2: SQL API Layer

#### 2.1 API Functions (`src/api.rs`) - 0% Active Coverage

The SQL function implementations (3,227 lines) are user-facing:

**Recommended tests:**
- `train()` function with various parameter combinations
- `predict()`, `predict_proba()`, `predict_text()` output correctness
- `embed()` function dimension validation
- `generate()` function text output
- Parameter validation and error messages
- Invalid input handling

### Priority 3: ORM Types

#### 3.1 Untested ORM Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `orm/project.rs` | 165 | Project CRUD operations |
| `orm/dataset.rs` | 686 | Dataset loading, preprocessing |
| `orm/algorithm.rs` | 158 | Algorithm enum parsing (49 variants) |
| `orm/task.rs` | 172 | Task type parsing and validation |
| `orm/sampling.rs` | 70 | Sampling strategy implementation |
| `orm/strategy.rs` | 38 | Deployment strategy logic |
| `orm/status.rs` | 32 | Status enum transitions |

**Recommended tests:**
- `Algorithm::from_str()` for all 49 variants
- `Task::from_str()` for all task types
- Project creation, lookup, and deletion
- Dataset train/test split correctness
- Sampling strategy randomness and reproducibility

### Priority 4: Existing Tests - Improve Coverage

#### 4.1 Metrics Module (Partial Coverage)

Current test only covers perfect prediction scenario. Add:
- Imperfect predictions with known F1 scores
- Binary classification metrics (precision, recall, AUC)
- Edge cases: all same class, single sample
- Different averaging methods (macro, micro, weighted)

#### 4.2 Context Module (Partial Coverage)

Current test only covers initialization. Add:
- `run()` function with successful/failing closures
- Error handling when context not initialized
- Concurrent access patterns

---

## Recommended Test Infrastructure

### 1. Add Dev Dependencies to Cargo.toml

```toml
[dev-dependencies]
tempfile = "3.1"
pretty_assertions = "1.4"
approx = "0.5"           # For floating-point comparisons
rstest = "0.18"          # Parameterized tests
criterion = "0.5"        # Benchmarking
```

### 2. Create Test Utilities Module

```
src/
├── test_utils/
│   ├── mod.rs           # Re-exports
│   ├── fixtures.rs      # Test data factories
│   ├── database.rs      # In-memory DB helpers
│   └── assertions.rs    # ML-specific assertions
```

**Suggested utilities:**
- `TestDatabase` - creates in-memory DuckDB with schema
- `make_iris_dataset()` - returns known test data
- `assert_prediction_accuracy()` - validates model predictions
- `assert_array_close()` - floating-point array comparison

### 3. Create Integration Tests Directory

```
tests/
├── train_test.rs        # End-to-end training tests
├── predict_test.rs      # Prediction pipeline tests
├── algorithms/
│   ├── linfa_test.rs
│   ├── xgboost_test.rs
│   └── sklearn_test.rs
└── common/
    └── mod.rs           # Shared test helpers
```

### 4. Sample Test Implementations

**Algorithm correctness test:**
```rust
#[test]
fn test_linear_regression_fits_linear_data() {
    // y = 2x + 1
    let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

    let model = LinfaBindings::fit_linear_regression(&x, &y).unwrap();
    let predictions = model.predict(&[3.0], 1, 0).unwrap();

    assert!((predictions[0] - 7.0).abs() < 0.1);
}
```

**Serialization round-trip test:**
```rust
#[test]
fn test_xgboost_serialization_preserves_predictions() {
    let model = train_xgboost_on_iris();
    let bytes = model.to_bytes().unwrap();
    let restored = XGBoostBindings::from_bytes(&bytes).unwrap();

    let original_pred = model.predict(&sample, 4, 3).unwrap();
    let restored_pred = restored.predict(&sample, 4, 3).unwrap();

    assert_eq!(original_pred, restored_pred);
}
```

---

## Test Coverage Goals

| Timeframe | Target | Focus Areas |
|-----------|--------|-------------|
| Immediate | 10% | Algorithm bindings, Model.fit() |
| Short-term | 30% | ORM types, Snapshot, Dataset |
| Medium-term | 60% | API layer, integration tests |
| Long-term | 80%+ | Edge cases, error handling, benchmarks |

---

## Specific Action Items

### Immediate Actions

1. **Add `[dev-dependencies]`** to Cargo.toml with testing utilities
2. **Uncomment and fix api.rs tests** - convert from `#[pg_test]` to `#[test]`
3. **Add unit tests for `linfa.rs`** - test LinearRegression, LogisticRegression, SVM
4. **Add unit tests for `algorithm.rs`** - test all 49 algorithm variants parse correctly
5. **Expand metrics.rs tests** - add imperfect prediction scenarios

### Short-term Actions

6. Create `tests/` directory for integration tests
7. Add XGBoost and LightGBM binding tests
8. Add Model serialization/deserialization tests
9. Add Dataset train/test split tests
10. Add Project CRUD tests

### Medium-term Actions

11. Add end-to-end SQL function tests
12. Add Python binding tests (sklearn, transformers)
13. Add performance benchmarks with Criterion
14. Set up CI/CD pipeline with test automation
15. Add property-based tests for edge cases

---

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|------------|------------|
| Silent algorithm regression | Critical | High | Add algorithm correctness tests |
| Data preprocessing bugs | High | High | Add snapshot/dataset tests |
| Serialization corruption | High | Medium | Add round-trip tests |
| SQL API failures | High | Medium | Add integration tests |
| Python interop issues | Medium | Medium | Add PyO3 binding tests |

---

## Conclusion

The current test coverage of ~0.02% represents significant risk for an ML system where correctness is critical. Priority should be given to:

1. **Algorithm bindings** - the core value proposition
2. **Model management** - training and prediction pipeline
3. **Data handling** - snapshots and datasets

The investment in test infrastructure will pay dividends in development velocity and confidence when making changes to this complex codebase.
