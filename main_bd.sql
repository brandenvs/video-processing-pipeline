-- Table to store visual analysis data
CREATE TABLE visual_analysis (
    id SERIAL PRIMARY KEY,
    frame_description TEXT[],
    license_plates TEXT[],
    scene_sentiment TEXT[],
    sentiment_justification TEXT[],
    people_nearby JSONB,
    risk_analysis TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- i need the visual analysis to be list of dicts, 
-- this is beacause ill need to identify multiple, license playes with times, scene setiment, people nearby , there descritpions , time , frame descrtipion should be a list of dict with time too

-- Table to store message logs associated with each analysis
CREATE TABLE analysis_messages (
    id SERIAL PRIMARY KEY,
    analysis_id INTEGER REFERENCES visual_analysis(id) ON DELETE CASCADE,
    role TEXT CHECK (role IN ('system', 'user')),
    content JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
