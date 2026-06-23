-- Database setup script for safety analysis reports
-- Run this script in your PostgreSQL database (via Supabase SQL editor or psql)

-- Enable pgvector extension (required for policy_embeddings_vs)
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- reports: one row per video analysis run
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.reports (
    id bigserial NOT NULL,
    report_id text NOT NULL,
    video_id text,
    video_duration double precision NOT NULL,
    video_captured_at timestamp with time zone NOT NULL,
    video_device_type text NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT now(),
    total_observations integer NOT NULL DEFAULT 0,
    low integer NOT NULL DEFAULT 0,
    medium integer NOT NULL DEFAULT 0,
    high integer NOT NULL DEFAULT 0,
    observations JSONB,
    CONSTRAINT reports_pkey PRIMARY KEY (id),
    CONSTRAINT reports_report_id_key UNIQUE (report_id)
) TABLESPACE pg_default;

-- Create index for efficient querying by capture time
CREATE INDEX IF NOT EXISTS reports_captured_at_idx 
ON public.reports USING btree (video_captured_at DESC) 
TABLESPACE pg_default;

-- Create index for efficient querying by device type
CREATE INDEX IF NOT EXISTS reports_device_type_idx 
ON public.reports USING btree (video_device_type) 
TABLESPACE pg_default;

-- Create index for efficient querying by total observations
CREATE INDEX IF NOT EXISTS reports_observations_idx 
ON public.reports USING btree (total_observations DESC) 
TABLESPACE pg_default;

-- ---------------------------------------------------------------------------
-- formal_reports: RAG-generated formal safety report per analysis run
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.formal_reports (
    id bigserial NOT NULL,
    report_id text NOT NULL,
    video_duration double precision NOT NULL,
    video_captured_at timestamp with time zone NOT NULL,
    video_device_type text NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT now(),
    total_observations integer NOT NULL DEFAULT 0,
    low integer NOT NULL DEFAULT 0,
    medium integer NOT NULL DEFAULT 0,
    high integer NOT NULL DEFAULT 0,
    observations JSONB,
    description text,                          -- RAG-generated formal report content
    CONSTRAINT formal_reports_pkey PRIMARY KEY (id),
    CONSTRAINT formal_reports_report_id_key UNIQUE (report_id)
) TABLESPACE pg_default;

-- Create indexes for formal_reports table
CREATE INDEX IF NOT EXISTS formal_reports_captured_at_idx 
ON public.formal_reports USING btree (video_captured_at DESC) 
TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS formal_reports_device_type_idx 
ON public.formal_reports USING btree (video_device_type) 
TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS formal_reports_observations_idx 
ON public.formal_reports USING btree (total_observations DESC) 
TABLESPACE pg_default;

-- ---------------------------------------------------------------------------
-- policy_embeddings: source-of-truth policy documents (read by rag_system.py)
--   Columns consumed: id, content, title, category
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.policy_embeddings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    content text NOT NULL,
    title text NOT NULL,
    category text NOT NULL,
    source text,
    created_at timestamp with time zone NOT NULL DEFAULT now()
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS policy_embeddings_category_idx
ON public.policy_embeddings USING btree (category)
TABLESPACE pg_default;

-- ---------------------------------------------------------------------------
-- policy_embeddings_vs: LangChain SupabaseVectorStore table
--   Used by rag_system.py with table_name="policy_embeddings_vs"
--   nomic-embed-text produces 768-dimensional vectors
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.policy_embeddings_vs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    content text NOT NULL,
    metadata jsonb DEFAULT '{}'::jsonb,
    embedding vector(768),
    created_at timestamp with time zone NOT NULL DEFAULT now()
) TABLESPACE pg_default;

CREATE INDEX IF NOT EXISTS policy_embeddings_vs_embedding_idx
ON public.policy_embeddings_vs
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100)
TABLESPACE pg_default;

-- ---------------------------------------------------------------------------
-- match_documents: RPC function used by LangChain SupabaseVectorStore
--   query_name="match_documents" in rag_system.py
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION match_documents(
    query_embedding vector(768),
    match_count int DEFAULT 10,
    filter jsonb DEFAULT '{}'::jsonb
)
RETURNS TABLE (
    id uuid,
    content text,
    metadata jsonb,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        pev.id,
        pev.content,
        pev.metadata,
        1 - (pev.embedding <=> query_embedding) AS similarity
    FROM public.policy_embeddings_vs pev
    ORDER BY pev.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Grant permissions (adjust as needed for your setup)
-- GRANT ALL PRIVILEGES ON TABLE public.reports TO your_app_user;
-- GRANT USAGE, SELECT ON SEQUENCE public.reports_id_seq TO your_app_user;
-- GRANT ALL PRIVILEGES ON TABLE public.formal_reports TO your_app_user;
-- GRANT USAGE, SELECT ON SEQUENCE public.formal_reports_id_seq TO your_app_user;
-- GRANT ALL PRIVILEGES ON TABLE public.policy_embeddings TO your_app_user;
-- GRANT ALL PRIVILEGES ON TABLE public.policy_embeddings_vs TO your_app_user;
-- GRANT EXECUTE ON FUNCTION match_documents TO your_app_user;
