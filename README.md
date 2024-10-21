# quackML
A duckDB extension implementing a full service AI/ML engine.

Bring your models to your data not the other way around.

Develop AI/ML models in pure sql with duckDB.

Current support for linear/logistic regression, xgboost, lightgbm, Huggingface intergration for text tasks and embedding.

Still in development.

# Examples

Fine tune gpt2

```
select * from finetune(
· 'IMDB Review Sentiment',
·     task => 'text_classification',
·     relation_name => 'quackml.glue_data',
·     y_column_name => 'class',
·     model_name => 'gpt2',
·     hyperparams => '{
·         "training_args": { "learning_rate": 2e-5,
·         "per_device_train_batch_size": 16,
·         "per_device_eval_batch_size": 16,
·         "num_train_epochs": 1,
·         "weight_decay": 0.01 }
·     }',
·     test_size => 0.5,
·     test_sampling => 'random'
· );
```



https://github.com/user-attachments/assets/47635f7f-3412-4401-aa9e-d4c00c4a8cec



