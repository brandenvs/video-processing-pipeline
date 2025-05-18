import psycopg2

conn = psycopg2.connect(
    dbname="postgres",
    user="postgres",
    host="localhost",
    password="postgres",
)
cursor = conn.cursor()

# Check if database exists
cursor.execute("SELECT 1 FROM pg_database WHERE datname='stadprin'")
exists = cursor.fetchone()

if not exists:
    cursor.execute("CREATE DATABASE stadprin")
    print("Created stadprin")
else:
    print("stadprin already exists")

cursor.close()
conn.close()

conn = psycopg2.connect(
    dbname="stadprin",
    user="postgres",
    host="localhost",
    password="postgres",
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

# Create form_documents table
cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS form_documents (
        id SERIAL PRIMARY KEY,
        document_name TEXT,
        document_path TEXT,
        video_path TEXT,
        fields_json JSONB,
        processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""
)
print("Created form_documents table...")

# @Branden - No commit was made after initial creation of tables.
conn.commit()
print("Tables created successfully!")

# Add pgai vectorconn.commit()("Created visual_analysis")
cursor.close()
conn.close()
