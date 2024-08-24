use std::{collections::HashMap, error::Error, path::Path};

use serde_json::Value;

use crate::orm::{
    ConversationDataset, Task, TextClassificationDataset, TextPairClassificationDataset,
};

pub fn finetune_text_classification(
    task: &Task,
    dataset: TextClassificationDataset,
    hyperparams: &Value,
    path: &Path,
    project_id: i64,
    model_id: i64,
) -> Result<HashMap<String, f64>, Box<dyn Error>> {
    // let task = task.to_string();
    // let hyperparams = serde_json::to_string(&hyperparams.0)?;

    // Python::with_gil(|py| -> Result<HashMap<String, f64>> {
    //     let tune = get_module!(PY_MODULE)
    //         .getattr(py, "finetune_text_classification")
    //         .format_traceback(py)?;
    //     let path = path.to_string_lossy();
    //     let output = tune
    //         .call1(
    //             py,
    //             (
    //                 &task,
    //                 &hyperparams,
    //                 path.as_ref(),
    //                 dataset.text_train,
    //                 dataset.text_test,
    //                 dataset.class_train,
    //                 dataset.class_test,
    //                 project_id,
    //                 model_id,
    //             ),
    //         )
    //         .format_traceback(py)?;

    //     output.extract(py).format_traceback(py)
    // })
    Err("Not implemented".into())
}

pub fn finetune_text_pair_classification(
    task: &Task,
    dataset: TextPairClassificationDataset,
    hyperparams: &Value,
    path: &Path,
    project_id: i64,
    model_id: i64,
) -> Result<HashMap<String, f64>, Box<dyn Error>> {
    // let task = task.to_string();
    // let hyperparams = serde_json::to_string(&hyperparams.0)?;

    // Python::with_gil(|py| -> Result<HashMap<String, f64>> {
    //     let tune = get_module!(PY_MODULE)
    //         .getattr(py, "finetune_text_pair_classification")
    //         .format_traceback(py)?;
    //     let path = path.to_string_lossy();
    //     let output = tune
    //         .call1(
    //             py,
    //             (
    //                 &task,
    //                 &hyperparams,
    //                 path.as_ref(),
    //                 dataset.text1_train,
    //                 dataset.text1_test,
    //                 dataset.text2_train,
    //                 dataset.text2_test,
    //                 dataset.class_train,
    //                 dataset.class_test,
    //                 project_id,
    //                 model_id,
    //             ),
    //         )
    //         .format_traceback(py)?;

    //     output.extract(py).format_traceback(py)
    // })
    Err("Not implemented".into())
}

pub fn finetune_conversation(
    task: &Task,
    dataset: ConversationDataset,
    hyperparams: &Value,
    path: &Path,
    project_id: i64,
    model_id: i64,
) -> Result<HashMap<String, f64>, Box<dyn Error>> {
    // let task = task.to_string();
    // let hyperparams = serde_json::to_string(&hyperparams.0)?;

    // Python::with_gil(|py| -> Result<HashMap<String, f64>> {
    //     let tune = get_module!(PY_MODULE)
    //         .getattr(py, "finetune_conversation")
    //         .format_traceback(py)?;
    //     let path = path.to_string_lossy();
    //     let output = tune
    //         .call1(
    //             py,
    //             (
    //                 &task,
    //                 &hyperparams,
    //                 path.as_ref(),
    //                 dataset.system_train,
    //                 dataset.user_test,
    //                 dataset.assistant_train,
    //                 dataset.system_test,
    //                 dataset.user_train,
    //                 dataset.assistant_test,
    //                 project_id,
    //                 model_id,
    //             ),
    //         )
    //         .format_traceback(py)?;

    //     output.extract(py).format_traceback(py)
    // })
    Err("Not implemented".into())
}
