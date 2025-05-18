import psycopg2
import json
from app.routers.database_service import DB_CONFIG

def setup_database():
    """Set up the database with the required tables for document processing"""
    
    # Connect to PostgreSQL (without specifying a database first)
    conn_params = DB_CONFIG.copy()
    db_name = conn_params.pop('database')
    
    # First connect to default 'postgres' database to check if our DB exists
    conn_params['database'] = 'postgres'
    
    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True  # Needed for creating database
        cursor = conn.cursor()
        
        # Check if the database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Creating database '{db_name}'...")
            cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"Database '{db_name}' created successfully!")
        else:
            print(f"Database '{db_name}' already exists.")
        
        cursor.close()
        conn.close()
        
        # Now connect to our actual database
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        print("Creating tables if they don't exist...")
        
        # Create form_fields table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS form_fields (
            id SERIAL PRIMARY KEY,
            document_name VARCHAR(255) NOT NULL,
            field_name VARCHAR(255) NOT NULL,
            field_value TEXT,
            field_length INTEGER,
            field_type VARCHAR(50),
            processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create form_documents table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS form_documents (
            id SERIAL PRIMARY KEY,
            document_name VARCHAR(255) NOT NULL,
            document_path VARCHAR(512),
            video_path VARCHAR(512),
            fields_json JSONB,
            processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create visual_analysis table with a single JSON column for analysis data
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS visual_analysis (
            id SERIAL PRIMARY KEY,
            source_path TEXT,
            analysis_data JSONB,
            batch_number TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create processing_results table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS processing_results (
            id SERIAL PRIMARY KEY,
            request_id VARCHAR(255),
            processor_type VARCHAR(100) NOT NULL,
            model VARCHAR(100),
            result_json JSONB,
            processing_time FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        conn.commit()
        print("Tables created successfully!")
        
        # Insert demo data into visual_analysis table
        sample_video_analysis = {
            "frame_activity": "Police officer inspecting a vehicle during a traffic stop",
            "objects_detected": "Police car, civilian vehicle, flashlight, ID card, driver's license",
            "cars_detected": [
                {
                    "license_plate": "ABC-123",
                    "color": "Blue",
                    "model": "Toyota Corolla"
                },
                {
                    "license_plate": "XYZ-789",
                    "color": "White",
                    "model": "Ford Police Interceptor"
                }
            ],
            "people_detected": [
                {
                    "estimated_height": "180cm",
                    "age": "35-45",
                    "race": "Caucasian",
                    "emotional_state": "Nervous",
                    "proximity": "Inside vehicle"
                },
                {
                    "estimated_height": "185cm",
                    "age": "30-40",
                    "race": "Caucasian",
                    "emotional_state": "Authoritative",
                    "proximity": "Standing by driver window"
                }
            ],
            "scene_sentiment": "neutral",
            "id_cards_detected": [
                {
                    "surname": "Smith",
                    "names": "John Robert",
                    "sex": "Male",
                    "nationality": "USA",
                    "identity_number": "123-45-6789",
                    "date_of_birth": "1985-06-15",
                    "country_of_birth": "USA",
                    "status": "Valid"
                }
            ]
        }
        
        # Check if demo data already exists
        cursor.execute("SELECT COUNT(*) FROM visual_analysis WHERE source_path = 'input/patrol_video.mp4'")
        count = cursor.fetchone()[0]
        
        if count == 0:
            # Insert the sample data
            cursor.execute("""
            INSERT INTO visual_analysis 
            (source_path, analysis_data, batch_number)
            VALUES (%s, %s, %s)
            """, (
                "input/patrol_video.mp4",
                json.dumps(sample_video_analysis),
                "BATCH001"
            ))
            conn.commit()
            print("Demo visual analysis data inserted successfully!")
        else:
            print("Demo visual analysis data already exists.")
        
        # List all tables in the database
        cursor.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        print("\nAvailable tables in the database:")
        for table in tables:
            print(f" - {table[0]}")
        
        cursor.close()
        conn.close()
        
        print("\nDatabase setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error setting up database: {str(e)}")
        return False

if __name__ == "__main__":
    setup_database()