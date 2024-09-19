use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// Agent types
#[derive(Serialize, Deserialize, Clone)]
pub enum AgentType {
    QLearning,
    DQN,
    PPO,
    PPOWithFeedback,
    Custom(String),
}

// Model definition
#[derive(Serialize, Deserialize, Clone)]
pub struct ModelDefinition {
    model_type: String,
    input_size: usize,
    output_size: usize,
    layers: Vec<LayerDefinition>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LayerDefinition {
    layer_type: String,
    units: usize,
    activation: String,
}

// Observation and Action Spaces
#[derive(Serialize, Deserialize, Clone)]
pub enum SpaceType {
    Box,
    Discrete,
    MultiDiscrete,
    MultiBinary,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Space {
    #[serde(rename = "type")]
    space_type: SpaceType,
    low: Option<Value>,
    high: Option<Value>,
    shape: Option<Vec<usize>>,
    n: Option<usize>,
    dtype: Option<String>,
}

// Environment structure
#[derive(Clone)]
pub struct Environment {
    name: String,
    id: Option<String>,
    observation_space: Space,
    action_space: Space,
    reward_range: (f64, f64),
    metadata: Value,
    max_episode_steps: usize,
    kwargs: Value,
}

// Feedback structure
pub struct Feedback {
    agent_name: String,
    episode_id: usize,
    step_id: usize,
    feedback_score: f64,
    feedback_comment: Option<String>,
}

// Agent structure
pub struct Agent {
    name: String,
    agent_type: AgentType,
    model: ModelDefinition,
    parameters: Value,
    library_name: String,
    rl_agent: Arc<dyn RLAgent + Send + Sync>,
}

pub trait RLAgent {
    fn train(
        &mut self,
        env_spec: &Environment,
        num_episodes: usize,
        parameters: &Value,
    ) -> Result<()>;
    fn predict_action(&self, state: &Value) -> Result<Value>;
}
