import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# First connect to default postgres database
try:
    conn = psycopg2.connect(
        dbname='postgres',  # Connect to default database first
        user='postgres',
        host='localhost',
        password='postgres',
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute("SELECT 1 FROM pg_database WHERE datname='stadprin'")
    exists = cursor.fetchone()
    
    if not exists:
        print("Creating database 'stadprin'...")
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier('stadprin')))
    else:
        print("Database 'stadprin' already exists, skipping creation")
    
    cursor.close()
    conn.close()
except Exception as e:
    print(f"Error during database creation: {e}")

# Connect to the stadprin database to create tables
try:
    conn = psycopg2.connect(
        dbname='stadprin',
        user='postgres',
        host='localhost',
        password='postgress',
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Create visual_analysis table
    cursor.execute("""
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
    """)
    print("Created visual_analysis table")
    
    # Create audio_analysis table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audio_analysis (
        id SERIAL PRIMARY KEY,
        summary TEXT,
        sentiment TEXT,
        tone TEXT,
        fixed_transcript TEXT,
        urgent BOOLEAN,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        source_path TEXT
    )
    """)
    print("Created audio_analysis table")
    
    cursor.close()
    conn.close()
    
    print("Database setup completed successfully")
except Exception as e:
    print(f"Error during table creation: {e}")
