-- Supabase pgvector setup for Face Recognition
-- Uses TABLE_PREFIX (default: facerec_) for your existing DB
-- Run this in Supabase SQL Editor: https://supabase.com/dashboard/project/YOUR_PROJECT/sql

-- Enable vector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create face embeddings table with prefix
CREATE TABLE IF NOT EXISTS facerec_embeddings (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    embedding VECTOR(512) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for faster similarity search
CREATE INDEX IF NOT EXISTS idx_facerec_embeddings 
    ON facerec_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Create RPC function for similarity search with prefix
CREATE OR REPLACE FUNCTION facerec_match_embeddings(
    query_embedding VECTOR(512),
    match_threshold FLOAT DEFAULT 0.45,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id BIGINT,
    name TEXT,
    embedding VECTOR(512),
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        f.id,
        f.name,
        f.embedding,
        1 - (f.embedding <=> query_embedding) AS similarity
    FROM facerec_embeddings f
    WHERE 1 - (f.embedding <=> query_embedding) >= match_threshold
    ORDER BY f.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Grant permissions
GRANT ALL ON facerec_embeddings TO anon;
GRANT ALL ON facerec_embeddings TO authenticated;
GRANT USAGE ON SEQUENCE facerec_embeddings_id_seq TO anon;
GRANT USAGE ON SEQUENCE facerec_embeddings_id_seq TO authenticated;

-- Grant execute on function
GRANT EXECUTE ON FUNCTION facerec_match_embeddings TO anon;
GRANT EXECUTE ON FUNCTION facerec_match_embeddings TO authenticated;
