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
        cars_detected TEXT,
        people_detected TEXT,
        scene_sentiment TEXT,
        id_cards_detected TEXT,
        batch_number TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source_path TEXT
    )
"""
)
print("Created visual_analysis")
cursor.close()
conn.close()
