use anyhow::{anyhow, Result};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::{bert, distilbert, falcon, gemma, gemma2, llama, mamba, mistral},
};
use duckdb::params;
use models::{
    BertTransformerModel, DistilBertTransformerModel, FalconTransformerModel,
    Gemma2TransformerModel, GemmaTransformerModel, LlamaTransformerModel, MambaTransformerModel,
    MistralTransformerModel,
};
use serde_json::Value;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokenizers::Tokenizer;

use candle_core::{DType, Device, Tensor};
use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::context;

use super::dump_model;

#[cfg(all(feature = "candle", not(feature = "python")))]
// Common type aliases
type ModelCache = HashMap<i64, Box<dyn TransformerModel>>;

// Global model cache
static MODEL_CACHE: Lazy<Mutex<ModelCache>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// Common inputs for all transformer models
#[derive(Debug)]
pub struct ModelInputs<'a> {
    pub input_ids: &'a Tensor,
    pub attention_mask: Option<&'a Tensor>,
    pub position_ids: Option<&'a Tensor>,
    pub seqlen_offset: Option<usize>,
}

/// Generation configuration with model-specific parameters
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_length: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub do_sample: bool,
    // Model-specific parameters stored in a HashMap
    pub model_specific_params: HashMap<String, f32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_length: 100,
            temperature: 0.7,
            top_p: 0.9,
            repetition_penalty: 1.1,
            do_sample: true,
            model_specific_params: HashMap::new(),
        }
    }
}

/// Generation output containing both logits and decoded text
#[derive(Debug)]
pub struct GenerationOutput {
    pub text: String,
    pub logits: Option<Tensor>,
    pub tokens: Vec<u32>,
}

/// Core trait for all transformer models
pub trait TransformerModel: Send + Sync {
    /// Forward pass common to all models
    fn forward(&mut self, inputs: &ModelInputs) -> Result<Tensor>;

    /// Model-specific generation logic
    fn generate(
        &mut self,
        input_ids: &Tensor,
        config: &GenerationConfig,
        tokenizer: &tokenizers::Tokenizer,
        logits_processor: &mut LogitsProcessor,
    ) -> Result<GenerationOutput>;

    /// Helper method for sampling tokens
    fn sample_token(&self, logits: &Tensor, top_p: f32) -> Result<u32> {
        // Implement top-p sampling
        unimplemented!("Default sampling not implemented")
    }
}

/// Text generation pipeline
pub struct TextGenerator {
    model: Arc<Box<dyn TransformerModel>>,
    tokenizer: tokenizers::Tokenizer,
    logits_processor: LogitsProcessor,
    config: GenerationConfig,
}

impl TextGenerator {
    pub fn new(
        model: Box<dyn TransformerModel>,
        tokenizer: tokenizers::Tokenizer,
        logits_processor: LogitsProcessor,
        config: Option<GenerationConfig>,
    ) -> Self {
        Self {
            model: Arc::new(model),
            tokenizer,
            logits_processor,
            config: config.unwrap_or_default(),
        }
    }

    pub fn generate(&mut self, prompt: &str) -> Result<String> {
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("{}", e))?;

        let input_ids = Tensor::new(tokens.get_ids(), &candle_core::Device::Cpu)?;

        let output = Arc::get_mut(&mut self.model)
            .ok_or_else(|| anyhow::anyhow!("Failed to get mutable reference to model"))?
            .generate(
                &input_ids,
                &self.config,
                &self.tokenizer,
                &mut self.logits_processor,
            )?;

        Ok(output.text)
    }
}

// Model implementations
pub mod models {
    use candle_transformers::models::{
        bert, distilbert, falcon, gemma, gemma2, llama, mamba, mistral,
    };

    use super::*;

    pub struct BertTransformerModel {
        inner: bert::BertModel,
    }

    impl BertTransformerModel {
        pub fn new(inner: bert::BertModel) -> Self {
            Self { inner }
        }
    }

    impl TransformerModel for BertTransformerModel {
        fn forward(&mut self, inputs: &ModelInputs) -> Result<Tensor> {
            self.inner
                .forward(
                    inputs.input_ids,
                    inputs.attention_mask.unwrap(),
                    inputs.position_ids,
                )
                .map_err(|e| anyhow!("{}", e))
        }

        fn generate(
            &mut self,
            input_ids: &Tensor,
            config: &GenerationConfig,
            tokenizer: &tokenizers::Tokenizer,
            logits_processor: &mut LogitsProcessor,
        ) -> Result<GenerationOutput> {
            // Implement BERT-specific generation logic
            unimplemented!("BERT generation not implemented")
        }
    }

    pub struct DistilBertTransformerModel {
        inner: distilbert::DistilBertModel,
    }

    impl DistilBertTransformerModel {
        pub fn new(inner: distilbert::DistilBertModel) -> Self {
            Self { inner }
        }
    }

    impl TransformerModel for DistilBertTransformerModel {
        fn forward(&mut self, inputs: &ModelInputs) -> Result<Tensor> {
            self.inner
                .forward(inputs.input_ids, inputs.attention_mask.unwrap())
                .map_err(|e| anyhow!("{}", e))
        }

        fn generate(
            &mut self,
            input_ids: &Tensor,
            config: &GenerationConfig,
            tokenizer: &tokenizers::Tokenizer,
            logits_processor: &mut LogitsProcessor,
        ) -> Result<GenerationOutput> {
            // Implement DistilBERT-specific generation logic
            unimplemented!("DistilBERT generation not implemented")
        }
    }

    pub struct FalconTransformerModel {
        inner: falcon::Falcon,
    }

    impl FalconTransformerModel {
        pub fn new(inner: falcon::Falcon) -> Self {
            Self { inner }
        }
    }

    impl TransformerModel for FalconTransformerModel {
        fn forward(&mut self, inputs: &ModelInputs) -> Result<Tensor> {
            self.inner
                .forward(inputs.input_ids)
                .map_err(|e| anyhow!("{}", e))
        }

        fn generate(
            &mut self,
            input_ids: &Tensor,
            config: &GenerationConfig,
            tokenizer: &tokenizers::Tokenizer,
            logits_processor: &mut LogitsProcessor,
        ) -> Result<GenerationOutput> {
            // Implement Falcon-specific generation logic
            unimplemented!("Falcon generation not implemented")
        }
    }

    pub struct GemmaTransformerModel {
        inner: gemma::Model,
    }

    impl GemmaTransformerModel {
        pub fn new(inner: gemma::Model) -> Self {
            Self { inner }
        }
    }

    impl TransformerModel for GemmaTransformerModel {
        fn forward(&mut self, inputs: &ModelInputs) -> Result<Tensor> {
            self.inner
                .forward(inputs.input_ids, inputs.seqlen_offset.unwrap_or(0))
                .map_err(|e| anyhow!("{}", e))
        }

        fn generate(
            &mut self,
            input_ids: &Tensor,
            config: &GenerationConfig,
            tokenizer: &tokenizers::Tokenizer,
            logits_processor: &mut LogitsProcessor,
        ) -> Result<GenerationOutput> {
            // Implement Gemma-specific generation logic
            unimplemented!("Gemma generation not implemented")
        }
    }

    pub struct Gemma2TransformerModel {
        inner: gemma2::Model,
    }

    impl Gemma2TransformerModel {
        pub fn new(inner: gemma2::Model) -> Self {
            Self { inner }
        }
    }

    impl TransformerModel for Gemma2TransformerModel {
        fn forward(&mut self, inputs: &ModelInputs) -> Result<Tensor> {
            self.inner
                .forward(inputs.input_ids, inputs.seqlen_offset.unwrap_or(0))
                .map_err(|e| anyhow!("{}", e))
        }

        fn generate(
            &mut self,
            input_ids: &Tensor,
            config: &GenerationConfig,
            tokenizer: &tokenizers::Tokenizer,
            logits_processor: &mut LogitsProcessor,
        ) -> Result<GenerationOutput> {
            // Implement Gemma2-specific generation logic
            unimplemented!("Gemma2 generation not implemented")
        }
    }

    pub struct LlamaTransformerModel {
        inner: llama::Llama,
        cache: llama::Cache,
    }

    impl LlamaTransformerModel {
        pub fn new(inner: llama::Llama, cache: llama::Cache) -> Self {
            Self { inner, cache }
        }
    }

    impl TransformerModel for LlamaTransformerModel {
        fn forward(&mut self, inputs: &ModelInputs) -> Result<Tensor> {
            self.inner
                .forward(
                    inputs.input_ids,
                    inputs.seqlen_offset.unwrap_or(0),
                    &mut self.cache,
                )
                .map_err(|e| anyhow!("{}", e))
        }

        fn generate(
            &mut self,
            input_ids: &Tensor,
            config: &GenerationConfig,
            tokenizer: &tokenizers::Tokenizer,
            logits_processor: &mut LogitsProcessor,
        ) -> Result<GenerationOutput> {
            let mut current_ids = input_ids.clone();
            let mut generated_tokens = current_ids.to_vec2::<u32>()?[0].clone();
            let mut output_text = String::new();

            // Llama-specific generation parameters
            let rope_scaling = config
                .model_specific_params
                .get("rope_scaling")
                .copied()
                .unwrap_or(1.0);

            for _ in 0..config.max_length {
                let inputs = ModelInputs {
                    input_ids: &current_ids,
                    attention_mask: None,
                    position_ids: None,
                    seqlen_offset: Some(generated_tokens.len()),
                };

                let logits = self.forward(&inputs)?;
                let next_token = logits_processor.sample(&logits)?;

                // Check for end of generation
                if next_token == tokenizer.token_to_id("</s>").unwrap_or(0) {
                    break;
                }

                generated_tokens.push(next_token);
                current_ids = Tensor::new(&[next_token], &candle_core::Device::Cpu)?;

                if let Ok(decoded) = tokenizer.decode(&[next_token], true) {
                    output_text.push_str(&decoded);
                }
            }

            Ok(GenerationOutput {
                text: output_text,
                logits: None,
                tokens: generated_tokens,
            })
        }
    }

    pub struct MambaTransformerModel {
        inner: mamba::Model,
        state: mamba::State,
    }

    impl MambaTransformerModel {
        pub fn new(inner: mamba::Model, state: mamba::State) -> Self {
            Self { inner, state }
        }
    }

    impl TransformerModel for MambaTransformerModel {
        fn forward(&mut self, inputs: &ModelInputs) -> Result<Tensor> {
            self.inner
                .forward(inputs.input_ids, &mut self.state)
                .map_err(|e| anyhow!("{}", e))
        }

        fn generate(
            &mut self,
            input_ids: &Tensor,
            config: &GenerationConfig,
            tokenizer: &tokenizers::Tokenizer,
            logits_processor: &mut LogitsProcessor,
        ) -> Result<GenerationOutput> {
            // Implement Mamba-specific generation logic
            unimplemented!("Mamba generation not implemented")
        }
    }

    pub struct MistralTransformerModel {
        inner: mistral::Model,
    }

    impl MistralTransformerModel {
        pub fn new(inner: mistral::Model) -> Self {
            Self { inner }
        }
    }

    impl TransformerModel for MistralTransformerModel {
        fn forward(&mut self, inputs: &ModelInputs) -> Result<Tensor> {
            self.inner
                .forward(inputs.input_ids, inputs.seqlen_offset.unwrap_or(0))
                .map_err(|e| anyhow!("{}", e))
        }

        fn generate(
            &mut self,
            input_ids: &Tensor,
            config: &GenerationConfig,
            tokenizer: &tokenizers::Tokenizer,
            logits_processor: &mut LogitsProcessor,
        ) -> Result<GenerationOutput> {
            let mut current_ids = input_ids.clone();
            let mut generated_tokens = current_ids.to_vec2::<u32>()?[0].clone();
            let mut output_text = String::new();

            // Mistral-specific sliding window attention
            let window_size = config
                .model_specific_params
                .get("sliding_window")
                .copied()
                .unwrap_or(4096.0) as usize;

            for _ in 0..config.max_length {
                let inputs = ModelInputs {
                    input_ids: &current_ids,
                    attention_mask: None,
                    position_ids: None,
                    seqlen_offset: Some(generated_tokens.len()),
                };

                let logits = self.forward(&inputs)?;
                let next_token = logits_processor.sample(&logits)?;

                if next_token == tokenizer.token_to_id("</s>").unwrap_or(0) {
                    break;
                }

                generated_tokens.push(next_token);

                // Maintain sliding window if needed
                if generated_tokens.len() > window_size {
                    generated_tokens =
                        generated_tokens[generated_tokens.len() - window_size..].to_vec();
                }

                current_ids = Tensor::new(&[next_token], &candle_core::Device::Cpu)?;

                if let Ok(decoded) = tokenizer.decode(&[next_token], true) {
                    output_text.push_str(&decoded);
                }
            }

            Ok(GenerationOutput {
                text: output_text,
                logits: None,
                tokens: generated_tokens,
            })
        }
    }
}

pub fn embed(
    transformer: &str,
    inputs: Vec<&str>,
    kwargs: &serde_json::Value,
) -> Result<Vec<Vec<f32>>> {
    // let tokenizer_params = FromPretrainedParameters::default();

    let tokenizer = Tokenizer::from_pretrained(transformer, None).map_err(|e| anyhow!("{}", e))?;
    let outputs = tokenizer.encode_batch(inputs, false).unwrap();
    let outputs = outputs
        .into_iter()
        .map(|x| x.get_ids().into_iter().map(|val| *val as f32).collect())
        .collect();
    Ok(outputs)
}

fn safetensor_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = vec![];
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(ext) = path.extension() {
            if ext == "safetensors" {
                paths.push(path);
            }
        }
    }
    Ok(paths)
}

fn load_model(model_id: i64, task: &str, dir: PathBuf) -> Result<Box<dyn TransformerModel>> {
    let config_reader = std::fs::File::open(dir.join("config.json"))?;

    let model_type = serde_json::from_reader::<_, Value>(config_reader)?
        .get("model_type")
        .map(|x| x.to_string())
        .unwrap_or_default();

    match model_type.as_str() {
        "bert" => {
            let config_reader = std::fs::File::open(dir.join("config.json"))?;
            let config: bert::Config = serde_json::from_reader(config_reader)?;

            let paths = safetensor_paths(&dir)?;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(paths.as_slice(), bert::DTYPE, &Device::Cpu)?
            };
            let inner_model = bert::BertModel::load(vb, &config)?;
            Ok(Box::new(BertTransformerModel::new(inner_model)))
        }
        "distilbert" => {
            let config_reader = std::fs::File::open(dir.join("config.json"))?;
            let config: distilbert::Config = serde_json::from_reader(config_reader)?;
            let paths = safetensor_paths(&dir)?;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(
                    paths.as_slice(),
                    distilbert::DTYPE,
                    &Device::Cpu,
                )?
            };
            let inner_model = distilbert::DistilBertModel::load(vb, &config)?;
            Ok(Box::new(DistilBertTransformerModel::new(inner_model)))
        }
        "falcon" => {
            let config_reader = std::fs::File::open(dir.join("config.json"))?;
            let config: falcon::Config = serde_json::from_reader(config_reader)?;
            let paths = safetensor_paths(&dir)?;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(paths.as_slice(), DType::F32, &Device::Cpu)?
            };
            let inner_model = falcon::Falcon::load(vb, config)?;
            Ok(Box::new(FalconTransformerModel::new(inner_model)))
        }
        "gemma" => {
            let config_reader = std::fs::File::open(dir.join("config.json"))?;
            let config: gemma::Config = serde_json::from_reader(config_reader)?;
            let paths = safetensor_paths(&dir)?;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(paths.as_slice(), DType::F32, &Device::Cpu)?
            };
            let inner_model = gemma::Model::new(false, &config, vb)?;
            Ok(Box::new(GemmaTransformerModel::new(inner_model)))
        }
        "gemma2" => {
            let config_reader = std::fs::File::open(dir.join("config.json"))?;
            let config: gemma2::Config = serde_json::from_reader(config_reader)?;
            let paths = safetensor_paths(&dir)?;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(paths.as_slice(), DType::F32, &Device::Cpu)?
            };
            let inner_model = gemma2::Model::new(false, &config, vb)?;
            Ok(Box::new(Gemma2TransformerModel::new(inner_model)))
        }
        "llama" => {
            let config_reader = std::fs::File::open(dir.join("config.json"))?;
            let config: llama::LlamaConfig = serde_json::from_reader(&config_reader)?;
            let config = config.into_config(false);
            let paths = safetensor_paths(&dir)?;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(paths.as_slice(), DType::F32, &Device::Cpu)?
            };
            let inner_model = llama::Llama::load(vb, &config)?;
            let cache = llama::Cache::new(true, DType::F32, &config, &Device::Cpu)?;
            Ok(Box::new(LlamaTransformerModel::new(inner_model, cache)))
        }
        "mamba" => {
            let config_reader = std::fs::File::open(dir.join("config.json"))?;
            let config: mamba::Config = serde_json::from_reader(config_reader)?;
            let paths = safetensor_paths(&dir)?;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(paths.as_slice(), DType::F32, &Device::Cpu)?
            };
            let inner_model = mamba::Model::new(&config, vb)?;
            let state = mamba::State::new(1, &config, DType::F32, &Device::Cpu)?;
            Ok(Box::new(MambaTransformerModel::new(inner_model, state)))
        }
        "mistral" => {
            let config_reader = std::fs::File::open(dir.join("config.json"))?;
            let config: mistral::Config = serde_json::from_reader(config_reader)?;
            let paths = safetensor_paths(&dir)?;
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(paths.as_slice(), DType::F32, &Device::Cpu)?
            };
            let inner_model = mistral::Model::new(&config, vb)?;
            Ok(Box::new(MistralTransformerModel::new(inner_model)))
        }
        _ => Err(anyhow!("Unsupported model type: {}", model_type.unwrap())),
    }
}

pub fn generate(
    model_id: i64,
    inputs: Vec<&str>,
    config: serde_json::Value,
) -> Result<Vec<String>> {
    let model = MODEL_CACHE.lock().unwrap().get(&model_id).cloned();
    let model = match model {
        Some(model) => model,
        None => {
            let mut dir = std::path::PathBuf::from("/tmp/quackml/models");
            dir.push(model_id.to_string());
            if !dir.exists() {
                dump_model(model_id, dir.clone())?;
            }
            let result = context::run(|conn| {
                conn.query_row(
                    "
                        SELECT task::TEXT
                        FROM quackml.projects
                        JOIN quackml.models
                            ON models.project_id = projects.id
                        WHERE models.id = $1
                        ",
                    params![model_id],
                    |row| row.get::<_, String>(0),
                )
                .map_err(|e| anyhow!("failed to get task: {e}"))
            });
            let task = result.expect("failed to get task");
            let model = load_model(model_id, &task, dir)?;
            MODEL_CACHE.lock().insert(model_id, model);
            model
        }
    };

    let tokenizer = Tokenizer::from_pretrained("gpt2", None).map_err(|e| anyhow!("{}", e))?;

    let mut logits_processor = LogitsProcessor::new(&config);
}

pub fn clear_gpu_cache(memory_usage: Option<f32>) -> Result<bool> {
    Ok(false)
}

pub fn load_dataset(
    name: &str,
    subset: Option<String>,
    limit: Option<usize>,
    kwargs: &serde_json::Value,
) -> Result<usize> {
    Ok(0)
}
