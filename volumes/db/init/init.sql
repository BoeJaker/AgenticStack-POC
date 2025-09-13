CREATE ROLE n8n_user WITH LOGIN PASSWORD 'your_secure_n8n_password';
CREATE ROLE webui_user WITH LOGIN PASSWORD 'your_secure_webui_password';

CREATE SCHEMA n8n AUTHORIZATION n8n_user;
CREATE SCHEMA webui AUTHORIZATION webui_user;

GRANT USAGE ON SCHEMA n8n TO n8n_user;
GRANT CREATE ON SCHEMA n8n TO n8n_user;

GRANT USAGE ON SCHEMA webui TO webui_user;
GRANT CREATE ON SCHEMA webui TO webui_user;


-- CREATE ROLE n8n_user WITH LOGIN PASSWORD 'your_secure_n8n_password';
-- CREATE DATABASE n8n OWNER n8n_user;

-- CREATE ROLE webui_user WITH LOGIN PASSWORD 'your_secure_webui_password';
-- CREATE DATABASE webui OWNER webui_user;

-- CREATE ROLE postgres WITH LOGIN PASSWORD 'your_secure_supabase_password';
-- CREATE DATABASE postgres OWNER postgres;
-- -- ----------------------------
-- -- 01_create_databases.sql
-- -- ----------------------------

-- -- Create roles
-- CREATE ROLE anon NOLOGIN;
-- CREATE ROLE authenticated NOLOGIN;
-- CREATE USER postgres WITH LOGIN PASSWORD 'your_secure_supabase_password';

-- -- N8N database
-- CREATE DATABASE n8n;
-- CREATE USER n8n_user WITH ENCRYPTED PASSWORD 'your_secure_n8n_password';
-- GRANT ALL PRIVILEGES ON DATABASE n8n TO n8n_user;

-- -- WebUI database
-- CREATE DATABASE webui;
-- CREATE USER webui_user WITH ENCRYPTED PASSWORD 'your_secure_webui_password';
-- GRANT ALL PRIVILEGES ON DATABASE webui TO webui_user;

-- -- MCP database
-- CREATE DATABASE mcp;
-- CREATE USER mcp_user WITH ENCRYPTED PASSWORD 'your_secure_mcp_password';
-- GRANT ALL PRIVILEGES ON DATABASE mcp TO mcp_user;

-- -- Tools database
-- CREATE DATABASE tools;
-- CREATE USER tools_user WITH ENCRYPTED PASSWORD 'your_secure_tools_password';
-- GRANT ALL PRIVILEGES ON DATABASE tools TO tools_user;

-- -- Agent database
-- CREATE DATABASE agent;
-- CREATE USER agent_user WITH ENCRYPTED PASSWORD 'your_secure_agent_password';
-- GRANT ALL PRIVILEGES ON DATABASE agent TO agent_user;


-- -- ----------------------------
-- -- 02_enable_extensions.sql
-- -- ----------------------------

-- -- Switch to agent database
-- \c agent;

-- -- Enable required extensions
-- CREATE EXTENSION IF NOT EXISTS vector;   -- pgvector
-- CREATE EXTENSION IF NOT EXISTS pg_trgm;
-- CREATE EXTENSION IF NOT EXISTS btree_gin;
-- CREATE EXTENSION IF NOT EXISTS btree_gist;


-- -- ----------------------------
-- -- 03_create_agent_schema.sql
-- -- ----------------------------

-- -- Agent Core Tables
-- CREATE TABLE IF NOT EXISTS interactions (
--     id BIGSERIAL PRIMARY KEY,
--     query TEXT NOT NULL,
--     response TEXT NOT NULL,
--     user_id TEXT NOT NULL,
--     session_id TEXT NOT NULL,
--     timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
--     complexity_score FLOAT,
--     model_used TEXT,
--     processing_time FLOAT,
--     actions_taken JSONB,
--     context JSONB
-- );

-- CREATE TABLE IF NOT EXISTS memory_embeddings (
--     id BIGSERIAL PRIMARY KEY,
--     content TEXT NOT NULL,
--     embedding vector(1536),
--     metadata JSONB,
--     user_id TEXT,
--     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
-- );

-- CREATE TABLE IF NOT EXISTS knowledge_entities (
--     id BIGSERIAL PRIMARY KEY,
--     name TEXT UNIQUE NOT NULL,
--     type TEXT NOT NULL,
--     properties JSONB,
--     embedding vector(1536),
--     confidence_score FLOAT DEFAULT 0.5,
--     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
--     updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
-- );

-- CREATE TABLE IF NOT EXISTS entity_relationships (
--     id BIGSERIAL PRIMARY KEY,
--     source_entity_id BIGINT REFERENCES knowledge_entities(id),
--     target_entity_id BIGINT REFERENCES knowledge_entities(id),
--     relationship_type TEXT NOT NULL,
--     strength FLOAT DEFAULT 0.5,
--     properties JSONB,
--     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
-- );

-- CREATE TABLE IF NOT EXISTS user_preferences (
--     id BIGSERIAL PRIMARY KEY,
--     user_id TEXT UNIQUE NOT NULL,
--     preferences JSONB,
--     interaction_style TEXT DEFAULT 'balanced',
--     complexity_preference FLOAT DEFAULT 0.5,
--     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
--     updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
-- );

-- CREATE TABLE IF NOT EXISTS proactive_thoughts (
--     id BIGSERIAL PRIMARY KEY,
--     thought TEXT NOT NULL,
--     trigger_context JSONB,
--     confidence_score FLOAT,
--     executed BOOLEAN DEFAULT FALSE,
--     created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
-- );

-- -- Indexes
-- CREATE INDEX idx_interactions_user_id ON interactions(user_id);
-- CREATE INDEX idx_interactions_session_id ON interactions(session_id);
-- CREATE INDEX idx_interactions_timestamp ON interactions(timestamp);
-- CREATE INDEX idx_memory_embeddings_user_id ON memory_embeddings(user_id);
-- CREATE INDEX idx_knowledge_entities_type ON knowledge_entities(type);
-- CREATE INDEX idx_entity_relationships_source ON entity_relationships(source_entity_id);
-- CREATE INDEX idx_entity_relationships_target ON entity_relationships(target_entity_id);

-- -- Vector similarity indexes (safe)
-- DO $$
-- BEGIN
--     IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
--         BEGIN
--             CREATE INDEX memory_embeddings_embedding_idx
--             ON memory_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
--         EXCEPTION WHEN OTHERS THEN
--             RAISE NOTICE 'Could not create ivfflat index; skipping.';
--         END;

--         BEGIN
--             CREATE INDEX knowledge_entities_embedding_idx
--             ON knowledge_entities USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
--         EXCEPTION WHEN OTHERS THEN
--             RAISE NOTICE 'Could not create ivfflat index; skipping.';
--         END;
--     ELSE
--         RAISE NOTICE 'pgvector not installed; skipping vector indexes.';
--     END IF;
-- END$$;
