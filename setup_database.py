import psycopg2
from app.routers.db_functions import DB_CONFIG

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