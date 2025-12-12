import os
import psycopg2

# Load DB credentials from environment to avoid committing secrets in source.
DB_NAME = os.getenv("POSTGRES_DB", "host")
DB_USER = os.getenv("POSTGRES_USER", "postgres")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")

if not DB_PASSWORD:
    raise RuntimeError(
        "Environment variable POSTGRES_PASSWORD not set. Create a .env from .env.example or export POSTGRES_PASSWORD."
    )

conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    host=DB_HOST,
    password=DB_PASSWORD,
)

cursor = conn.cursor()

# Check if database exists
cursor.execute("SELECT 1 FROM pg_database WHERE datname='host'")
exists = cursor.fetchone()

if not exists:
    cursor.execute("CREATE DATABASE host")
    print("Created host")
else:
    print("host already exists")

cursor.close()
conn.close()

conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    host=DB_HOST,
    password=DB_PASSWORD,
)

cursor = conn.cursor()
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS visual_analysis (
        id SERIAL PRIMARY KEY,
        frame_activity TEXT,
        objects_detected TEXT,
        cars_detected JSONB,
        people_detected JSONB,
        scene_sentiment TEXT,
        id_cards_detected JSONB,
        batch_number TEXT,
        source_path TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""
)


""" Document Tables Below"""


# Create form_fields table
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS form_fields (
        id SERIAL PRIMARY KEY,
        document_name TEXT,
        field_name TEXT,
        field_value TEXT,
        field_length INTEGER,
        field_type TEXT,
        processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""
)
print("Created form_fields table...")

# Create form_documents table with enhanced schema
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS form_documents (
        id SERIAL PRIMARY KEY,
        client_id TEXT,
        document_title TEXT,
        document_type TEXT,
        is_active BOOLEAN DEFAULT TRUE,
        schema_fields JSONB,
        generated_system_prompt TEXT,
        video_processing_prompt JSONB,
        document_path TEXT,
        video_path TEXT,
        source_path TEXT,
        created_by TEXT,
        created_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""
)
print("Created form_documents table...")

# @Branden - No commit was made after initial creation of tables.
conn.commit()
print("Tables created successfully!")

cursor.close()
conn.close()
