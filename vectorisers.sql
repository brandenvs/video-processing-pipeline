-- This is just a mockup vectorizer with search
SELECT ai.create_vectorizer(
     'processing_results'::regclass,
     destination => 'processing_results_embeddings',
     embedding => ai.embedding_ollama('all-minilm', 384),
     chunking => ai.chunking_recursive_character_text_splitter('result_json')
);

SELECT answer->>'response' as summary
FROM ai.ollama_generate('tinyllama', 
'Summarize the following and output in a detailed police report: '|| (SELECT text FROM processing_results WHERE created_at like 'pgai%')) as answer;

-- Create (actual table)
CREATE TABLE processing_results (
    id SERIAL PRIMARY KEY,
    request_id TEXT NOT NULL,
    processor_type TEXT NOT NULL,
    model TEXT NOT NULL,
    result_json JSONB NOT NULL,
    processing_time TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

