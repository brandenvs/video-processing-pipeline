import psycopg2
from app.routers.db_functions import DB_CONFIG

def check_database_connection():
    """Check if we can connect to the database"""
    try:
        print("Attempting to connect to database with these settings:")
        print(f"Host: {DB_CONFIG['host']}")
        print(f"Port: {DB_CONFIG['port']}")
        print(f"Database: {DB_CONFIG['database']}")
        print(f"User: {DB_CONFIG['user']}")
        print(f"Password: {'*' * len(DB_CONFIG['password'])}")
        
        # Try to connect
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Check if we can execute a query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"Successfully connected to PostgreSQL. Version: {version[0]}")
        
        # Check if our tables exist
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        print("\nExisting tables:")
        for table in tables:
            print(f" - {table[0]}")
        
        # Close connection
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Database connection error: {str(e)}")
        return False

if __name__ == "__main__":
    check_database_connection()