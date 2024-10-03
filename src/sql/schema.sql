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

CREATE TYPE status AS ENUM (
		'pending',
    'in_progress',
		'running',
		'sucessful',
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
-- CREATE INDEX IF NOT EXISTS models_project_id_idx ON quackml.models(project_id);
-- CREATE INDEX IF NOT EXISTS models_snapshot_id_idx ON quackml.models(snapshot_id);

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
--- Model Logs for tracking model performance
---
CREATE SEQUENCE IF NOT EXISTS quackml.model_logs_id_seq START 1;
CREATE TABLE IF NOT EXISTS quackml.logs(
    id BIGINT PRIMARY KEY DEFAULT nextval('quackml.model_logs_id_seq'),
    model_id BIGINT NOT NULL REFERENCES quackml.models(id),
    project_id BIGINT NOT NULL REFERENCES quackml.projects(id),
    logs JSON NOT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp(),
    updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT get_current_timestamp()
);
CREATE INDEX IF NOT EXISTS logs_model_id_idx ON quackml.logs(model_id);

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

-- 1. Overview View
---
--- Quick status check on the system.
---
DROP VIEW IF EXISTS quackml.overview;
CREATE VIEW quackml.overview AS
SELECT
	   p.name,
	   d.created_at AS deployed_at,
       p.task,
       m.algorithm,
       s.relation_name,
       s.y_column_name,
       s.test_sampling,
       s.test_size
FROM quackml.projects p
INNER JOIN quackml.models m ON p.id = m.project_id
INNER JOIN quackml.deployments d ON d.project_id = p.id
AND d.model_id = m.id
INNER JOIN quackml.snapshots s ON s.id = m.snapshot_id
ORDER BY d.created_at DESC;


---
--- List details of trained models.
---
DROP VIEW IF EXISTS quackml.trained_models;
CREATE VIEW quackml.trained_models AS
SELECT
	m.id,
	p.name,
	p.task,
	m.algorithm,
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
	ORDER BY project_id, created_at desc
) d ON d.model_id = m.id
ORDER BY m.created_at DESC;


---
--- List details of deployed models.
---
DROP VIEW IF EXISTS quackml.deployed_models;
CREATE VIEW quackml.deployed_models AS
SELECT
	m.id,
	p.name,
	p.task,
	m.algorithm,
	d.created_at as deployed_at
FROM quackml.projects p
INNER JOIN (
	SELECT DISTINCT ON(project_id)
		project_id, model_id, created_at
	FROM quackml.deployments
	ORDER BY project_id, created_at desc
) d ON d.project_id = p.id
INNER JOIN quackml.models m ON m.id = d.model_id
ORDER BY p.name ASC;
