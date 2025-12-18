-- Migration 001: Initial schema
-- Toolhub unified knowledge store schema
-- Requires: pgvector extension, pg_trgm extension

-- Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Sources: documentation sources (repos, websites, llms.txt files)
CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    canonical_url TEXT NOT NULL,
    source_type TEXT NOT NULL,  -- 'github', 'website', 'llmstxt'
    collection TEXT NOT NULL DEFAULT 'default',
    tags TEXT[] NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',  -- 'pending', 'crawling', 'indexed', 'failed'
    sha TEXT,  -- content hash for change detection
    fetched_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(canonical_url, collection)
);

-- Source artifacts: S3 key pointers for raw/extracted/manifest
CREATE TABLE IF NOT EXISTS source_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,  -- 'raw', 'extracted', 'manifest'
    s3_key TEXT NOT NULL,
    content_type TEXT,
    size_bytes BIGINT,
    sha256 TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(source_id, kind)
);

-- Chunks: chunked content with embeddings and FTS
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    heading TEXT,  -- current heading
    heading_path TEXT,  -- full path like 'Security > OAuth2 > Scopes'
    source_file TEXT,  -- original file path within source
    is_code BOOLEAN NOT NULL DEFAULT FALSE,
    chunk_index INTEGER NOT NULL,  -- position within source
    model_id TEXT NOT NULL,  -- embedding model used
    embedding vector(384),  -- all-MiniLM-L6-v2 produces 384-dim vectors
    search_vector tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(heading, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(heading_path, '')), 'A') ||
        setweight(to_tsvector('english', content), 'B')
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Entity types: JSON Schema registry with versioning
CREATE TABLE IF NOT EXISTS entity_types (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type_key TEXT NOT NULL UNIQUE,  -- 'competitor', 'wisdom', etc.
    json_schema JSONB NOT NULL,
    schema_version INTEGER NOT NULL DEFAULT 1,
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Entities: typed items with JSONB profiles
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type_key TEXT NOT NULL REFERENCES entity_types(type_key) ON DELETE RESTRICT,
    name TEXT NOT NULL,
    profile JSONB NOT NULL DEFAULT '{}',
    tags TEXT[] NOT NULL DEFAULT '{}',
    collection TEXT NOT NULL DEFAULT 'default',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(type_key, name, collection)
);

-- Evidence: links entity claims to source chunks
CREATE TABLE IF NOT EXISTS evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    field_path TEXT NOT NULL,  -- 'features', 'funding.total', 'pricing.notes[0]'
    chunk_id UUID REFERENCES chunks(id) ON DELETE SET NULL,
    source_id UUID REFERENCES sources(id) ON DELETE SET NULL,
    quote TEXT,  -- exact quote from source
    locator TEXT,  -- additional location info (line number, selector, etc.)
    confidence REAL,  -- 0.0-1.0 extraction confidence
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT evidence_has_reference CHECK (chunk_id IS NOT NULL OR source_id IS NOT NULL)
);

-- Indexes

-- Sources
CREATE INDEX IF NOT EXISTS idx_sources_collection ON sources(collection);
CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status);
CREATE INDEX IF NOT EXISTS idx_sources_tags ON sources USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_sources_fetched_at ON sources(fetched_at);

-- Source artifacts
CREATE INDEX IF NOT EXISTS idx_source_artifacts_source_id ON source_artifacts(source_id);

-- Chunks: vector similarity search (IVFFLAT for performance)
-- Note: IVFFLAT requires data to exist; create after initial data load
-- For small datasets, use exact search (no index)
CREATE INDEX IF NOT EXISTS idx_chunks_source_id ON chunks(source_id);
CREATE INDEX IF NOT EXISTS idx_chunks_search_vector ON chunks USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_chunks_heading ON chunks(heading);

-- Entity types
-- Note: type_key already has a UNIQUE constraint which creates an implicit index

-- Entities
CREATE INDEX IF NOT EXISTS idx_entities_type_key ON entities(type_key);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_collection ON entities(collection);
CREATE INDEX IF NOT EXISTS idx_entities_tags ON entities USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_entities_profile ON entities USING GIN(profile);

-- Evidence
CREATE INDEX IF NOT EXISTS idx_evidence_entity_id ON evidence(entity_id);
CREATE INDEX IF NOT EXISTS idx_evidence_chunk_id ON evidence(chunk_id);
CREATE INDEX IF NOT EXISTS idx_evidence_source_id ON evidence(source_id);
CREATE INDEX IF NOT EXISTS idx_evidence_field_path ON evidence(field_path);

-- Trigger for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE OR REPLACE TRIGGER update_sources_updated_at
    BEFORE UPDATE ON sources
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE TRIGGER update_entity_types_updated_at
    BEFORE UPDATE ON entity_types
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE OR REPLACE TRIGGER update_entities_updated_at
    BEFORE UPDATE ON entities
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Note: IVFFLAT index for chunks.embedding should be created after initial data load:
-- CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
-- The number of lists should be sqrt(n) where n is the number of rows.
