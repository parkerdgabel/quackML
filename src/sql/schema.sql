-- Create the schema for the QuackML database
CREATE SCHEMA IF NOT EXISTS quackml;

-- Task enum for basic task types
CREATE TYPE task AS ENUM (
    'regression',
    'classification',
    'decompisition',
    'clustering',
    'question_answering',
    'summarization',
    'translation',
    'text_classification',
    'text_generation',
    'text2txt',
    'embedding',
    'text_pair_classification',
    'conversation'
);

-- Sampling enum for basic sampling types
CREATE TYPE sampling AS ENUM (
    'random',
    'stratified',
    'time_series',
    'last'
);

-- Strategy enum for basic deployment strategies
CREATE TYPE strategy AS ENUM (
    'new_score',
    'best_score',
    'most_recent',
    'rollback',
    'specific'
);

CREATE TYPE STATUS AS ENUM (
		'pending',
    'in_progress',
		'running',
		'completed',
		'failed'
);

-- Projects Table to organize work
CREATE SEQUENCE IF NOT EXISTS quackml.projects_id_seq START 1;
CREATE TABLE IF NOT EXISTS quackml.projects (
	id BIGINT PRIMARY KEY DEFAULT nextval('quackml.projects_id_seq'),
	name TEXT NOT NULL,
	task task NOT NULL,
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp(),
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp(),
    deleted_at TIMESTAMP WITHOUT TIME ZONE
);
CREATE UNIQUE INDEX IF NOT EXISTS projects_name_idx ON quackml.projects(name);

---
--- Snapshots freeze data for training
---
CREATE SEQUENCE IF NOT EXISTS quackml.snapshots_id_seq START 1;
CREATE TABLE IF NOT EXISTS quackml.snapshots(
	id BIGINT PRIMARY KEY DEFAULT nextval('quackml.snapshots_id_seq'),
	relation_name TEXT NOT NULL,
	y_column_name TEXT[],
	test_size FLOAT4 NOT NULL,
	test_sampling sampling NOT NULL,
	status TEXT NOT NULL,
	columns TEXT,
	analysis TEXT,
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp(),
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp(),
	materialized BOOLEAN DEFAULT false
);

---
--- Models save the learned parameters
---
CREATE SEQUENCE IF NOT EXISTS quackml.models_id_seq START 1;
CREATE TABLE IF NOT EXISTS quackml.models(
	id BIGINT PRIMARY KEY DEFAULT nextval('quackml.models_id_seq'),
	project_id BIGINT NOT NULL REFERENCES quackml.projects(id),
	snapshot_id BIGINT REFERENCES quackml.snapshots(id),
	num_features INT NOT NULL,
	algorithm TEXT NOT NULL,
	hyperparams JSON NOT NULL,
	status TEXT NOT NULL,
	metrics JSON,
	search JSON,
	search_params JSON NOT NULL,
	search_args JSON NOT NULL,
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp(),
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp()
);

CREATE INDEX IF NOT EXISTS models_project_id_idx ON quackml.models(project_id);
CREATE INDEX IF NOT EXISTS models_snapshot_id_idx ON quackml.models(snapshot_id);

---
--- Deployments determine which model is live
---
CREATE SEQUENCE IF NOT EXISTS quackml.deployments_id_seq START 1;
CREATE TABLE IF NOT EXISTS quackml.deployments(
	id BIGINT PRIMARY KEY DEFAULT nextval('quackml.deployments_id_seq'),
	project_id BIGINT NOT NULL REFERENCES quackml.projects(id),
	model_id BIGINT NOT NULL REFERENCES quackml.models(id),
	strategy strategy NOT NULL,
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp(),
);
CREATE INDEX IF NOT EXISTS deployments_project_id_created_at_idx ON quackml.deployments(project_id);
CREATE INDEX IF NOT EXISTS deployments_model_id_created_at_idx ON quackml.deployments(model_id);

---
--- Distribute serialized models consistently for HA
---
CREATE SEQUENCE IF NOT EXISTS quackml.files_id_seq START 1;
CREATE TABLE IF NOT EXISTS quackml.files(
	id BIGINT PRIMARY KEY DEFAULT nextval('quackml.files_id_seq'),
	model_id BIGINT NOT NULL REFERENCES quackml.models(id),
	path TEXT NOT NULL,
	part INTEGER NOT NULL,
	created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp(),
	updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp(),
	data BYTEA NOT NULL,
);
CREATE UNIQUE INDEX IF NOT EXISTS files_model_id_path_part_idx ON quackml.files(model_id, path, part);

-- LLMs Table for Large Language Models
CREATE SEQUENCE IF NOT EXISTS quackml.llms_id_seq START 1;

CREATE TABLE IF NOT EXISTS quackml.llms (
    id BIGINT PRIMARY KEY DEFAULT nextval('quackml.llms_id_seq'),
    project_id BIGINT NOT NULL REFERENCES quackml.projects(id),
    snapshot_id BIGINT REFERENCES quackml.snapshots(id),
    base_model TEXT NOT NULL,         -- e.g., 'gpt2', 'bert'
    hyperparams JSON NOT NULL,
    tokenizer_details JSON,           -- Specific to LLMs
    status STATUS NOT NULL,
    metrics JSON,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp(),
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp()
);

CREATE INDEX IF NOT EXISTS llms_project_id_idx ON quackml.llms(project_id);
CREATE INDEX IF NOT EXISTS llms_snapshot_id_idx ON quackml.llms(snapshot_id);

-- Deployments Table for LLMs
CREATE SEQUENCE IF NOT EXISTS quackml.llm_deployments_id_seq START 1;

CREATE TABLE IF NOT EXISTS quackml.llm_deployments (
    id BIGINT PRIMARY KEY DEFAULT nextval('quackml.llm_deployments_id_seq'),
    project_id BIGINT NOT NULL REFERENCES quackml.projects(id),
    llm_id BIGINT NOT NULL REFERENCES quackml.llms(id),
    strategy strategy NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp()
);

CREATE INDEX IF NOT EXISTS llm_deployments_project_id_created_at_idx ON quackml.llm_deployments(project_id);
CREATE INDEX IF NOT EXISTS llm_deployments_llm_id_created_at_idx ON quackml.llm_deployments(llm_id);

-- Files Table for LLMs
CREATE SEQUENCE IF NOT EXISTS quackml.llm_files_id_seq START 1;

CREATE TABLE IF NOT EXISTS quackml.llm_files (
    id BIGINT PRIMARY KEY DEFAULT nextval('quackml.llm_files_id_seq'),
    llm_id BIGINT NOT NULL REFERENCES quackml.llms(id),
    path TEXT NOT NULL,
    part INTEGER NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp(),
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp(),
    data BYTEA NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS llm_files_llm_id_path_part_idx ON quackml.llm_files(llm_id, path, part);

-- Environments Table for RL Agents
CREATE SEQUENCE IF NOT EXISTS quackml.environments_id_seq START 1;

CREATE TABLE IF NOT EXISTS quackml.environments (
    id BIGINT PRIMARY KEY DEFAULT nextval('quackml.environments_id_seq'),
    name TEXT NOT NULL,
    observation_space JSON NOT NULL,  -- Defines the observation space
    action_space JSON NOT NULL,       -- Defines the action space
    reward_range JSON,                -- Optional: Defines the reward range
    metadata JSON,                    -- Additional environment metadata
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp(),
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp()
);

CREATE UNIQUE INDEX IF NOT EXISTS environments_name_idx ON quackml.environments(name);

-- RL Agents Table
CREATE SEQUENCE IF NOT EXISTS quackml.rl_agents_id_seq START 1;

CREATE TABLE IF NOT EXISTS quackml.rl_agents (
    id BIGINT PRIMARY KEY DEFAULT nextval('quackml.rl_agents_id_seq'),
    project_id BIGINT NOT NULL REFERENCES quackml.projects(id),
    snapshot_id BIGINT REFERENCES quackml.snapshots(id),
    environment_id BIGINT NOT NULL REFERENCES quackml.environments(id),
    algorithm TEXT NOT NULL,          -- e.g., 'ppo', 'dqn'
    hyperparams JSON NOT NULL,
    status STATUS NOT NULL,
    metrics JSON,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp(),
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp()
);

CREATE INDEX IF NOT EXISTS rl_agents_project_id_idx ON quackml.rl_agents(project_id);
CREATE INDEX IF NOT EXISTS rl_agents_snapshot_id_idx ON quackml.rl_agents(snapshot_id);
CREATE INDEX IF NOT EXISTS rl_agents_environment_id_idx ON quackml.rl_agents(environment_id);

-- Deployments Table for RL Agents
CREATE SEQUENCE IF NOT EXISTS quackml.rl_deployments_id_seq START 1;

CREATE TABLE IF NOT EXISTS quackml.rl_deployments (
    id BIGINT PRIMARY KEY DEFAULT nextval('quackml.rl_deployments_id_seq'),
    project_id BIGINT NOT NULL REFERENCES quackml.projects(id),
    rl_agent_id BIGINT NOT NULL REFERENCES quackml.rl_agents(id),
    strategy strategy NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp()
);

CREATE INDEX IF NOT EXISTS rl_deployments_project_id_created_at_idx ON quackml.rl_deployments(project_id);
CREATE INDEX IF NOT EXISTS rl_deployments_rl_agent_id_created_at_idx ON quackml.rl_deployments(rl_agent_id);

-- Files Table for RL Agents
CREATE SEQUENCE IF NOT EXISTS quackml.rl_files_id_seq START 1;

CREATE TABLE IF NOT EXISTS quackml.rl_files (
    id BIGINT PRIMARY KEY DEFAULT nextval('quackml.rl_files_id_seq'),
    rl_agent_id BIGINT NOT NULL REFERENCES quackml.rl_agents(id),
    path TEXT NOT NULL,
    part INTEGER NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp(),
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp(),
    data BYTEA NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS rl_files_rl_agent_id_path_part_idx ON quackml.rl_files(rl_agent_id, path, part);

-- 1. Overview View
DROP VIEW IF EXISTS quackml.overview;
CREATE VIEW quackml.overview AS
SELECT
    p.name AS project_name,
    p.task,
    m.algorithm AS model_algorithm,
    m.status AS model_status,
    m.created_at AS model_created_at,
    'StandardML' AS model_type
FROM quackml.projects p
INNER JOIN quackml.models m ON p.id = m.project_id
UNION ALL
SELECT
    p.name AS project_name,
    p.task,
    l.base_model AS model_algorithm,
    l.status AS model_status,
    l.created_at AS model_created_at,
    'LLM' AS model_type
FROM quackml.projects p
INNER JOIN quackml.llms l ON p.id = l.project_id
UNION ALL
SELECT
    p.name AS project_name,
    p.task,
    r.algorithm AS model_algorithm,
    r.status AS model_status,
    r.created_at AS model_created_at,
    'RLAgent' AS model_type
FROM quackml.projects p
INNER JOIN quackml.rl_agents r ON p.id = r.project_id
ORDER BY model_created_at DESC;

-- 2. Trained Models Views

-- Trained Models View for Standard ML
DROP VIEW IF EXISTS quackml.trained_models_standard_ml;
CREATE VIEW quackml.trained_models_standard_ml AS
SELECT
    m.id,
    p.name AS project_name,
    p.task,
    m.algorithm,
    m.hyperparams,
    m.created_at,
    s.test_sampling,
    s.test_size,
    d.model_id IS NOT NULL AS deployed
FROM quackml.projects p
INNER JOIN quackml.models m ON p.id = m.project_id
INNER JOIN quackml.snapshots s ON s.id = m.snapshot_id
LEFT JOIN (
    SELECT DISTINCT ON(project_id)
        project_id, model_id, created_at
    FROM quackml.deployments
    ORDER BY project_id, created_at DESC
) d ON d.model_id = m.id
ORDER BY m.created_at DESC;

-- Trained Models View for LLMs
DROP VIEW IF EXISTS quackml.trained_models_llms;
CREATE VIEW quackml.trained_models_llms AS
SELECT
    l.id,
    p.name AS project_name,
    p.task,
    l.base_model AS algorithm,
    l.hyperparams,
    l.tokenizer_details,
    l.created_at,
    s.test_sampling,
    s.test_size,
    d.llm_id IS NOT NULL AS deployed
FROM quackml.projects p
INNER JOIN quackml.llms l ON p.id = l.project_id
INNER JOIN quackml.snapshots s ON s.id = l.snapshot_id
LEFT JOIN (
    SELECT DISTINCT ON(project_id)
        project_id, llm_id, created_at
    FROM quackml.llm_deployments
    ORDER BY project_id, created_at DESC
) d ON d.llm_id = l.id
ORDER BY l.created_at DESC;

-- Trained Models View for RL Agents
DROP VIEW IF EXISTS quackml.trained_models_rl_agents;
CREATE VIEW quackml.trained_models_rl_agents AS
SELECT
    r.id,
    p.name AS project_name,
    p.task,
    r.algorithm,
    r.hyperparams,
    r.created_at,
    s.test_sampling,
    s.test_size,
    d.rl_agent_id IS NOT NULL AS deployed
FROM quackml.projects p
INNER JOIN quackml.rl_agents r ON p.id = r.project_id
INNER JOIN quackml.snapshots s ON s.id = r.snapshot_id
LEFT JOIN (
    SELECT DISTINCT ON(project_id)
        project_id, rl_agent_id, created_at
    FROM quackml.rl_deployments
    ORDER BY project_id, created_at DESC
) d ON d.rl_agent_id = r.id
ORDER BY r.created_at DESC;

-- 3. Deployed Models Views

-- Deployed Models View for Standard ML
DROP VIEW IF EXISTS quackml.deployed_models_standard_ml;
CREATE VIEW quackml.deployed_models_standard_ml AS
SELECT
    m.id,
    p.name AS project_name,
    p.task,
    m.algorithm,
    d.created_at AS deployed_at
FROM quackml.projects p
INNER JOIN quackml.deployments d ON d.project_id = p.id
INNER JOIN quackml.models m ON m.id = d.model_id
ORDER BY p.name ASC;

-- Deployed Models View for LLMs
DROP VIEW IF EXISTS quackml.deployed_models_llms;
CREATE VIEW quackml.deployed_models_llms AS
SELECT
    l.id,
    p.name AS project_name,
    p.task,
    l.base_model AS algorithm,
    d.created_at AS deployed_at
FROM quackml.projects p
INNER JOIN quackml.llm_deployments d ON d.project_id = p.id
INNER JOIN quackml.llms l ON l.id = d.llm_id
ORDER BY p.name ASC;

-- Deployed Models View for RL Agents
DROP VIEW IF EXISTS quackml.deployed_models_rl_agents;
CREATE VIEW quackml.deployed_models_rl_agents AS
SELECT
    r.id,
    p.name AS project_name,
    p.task,
    r.algorithm,
    d.created_at AS deployed_at
FROM quackml.projects p
INNER JOIN quackml.rl_deployments d ON d.project_id = p.id
INNER JOIN quackml.rl_agents r ON r.id = d.rl_agent_id
ORDER BY p.name ASC;

