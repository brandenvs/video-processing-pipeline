from datetime import datetime
import os
import psycopg2
from pydantic import Json

# DB configuration that works with both Docker and local development
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "database": os.getenv("POSTGRES_DB", "stadprin"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
    "port": os.getenv("POSTGRES_PORT", "5432")
}

def processed_video(result, processor_type):
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()
    
    # Generate a unique ID for this request
    request_id = f"{processor_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # SQL for inserting the result
    sql = """
    INSERT INTO processing_results 
    (request_id, processor_type, model, result_json, processing_time, created_at)
    VALUES (%s, %s, %s, %s, %s, %s)
    RETURNING id;
    """
    
    # Format result data for insertion
    values = (
        request_id,
        processor_type,
        result.get('model', 'unknown'),
        Json({"result": result.get('result', '')}),
        result.get('time_to_process', 0.0),
        datetime.now()
    ) 
    print(f'{datetime.now()}')
    
    cursor.execute(sql, values)
    db_id = cursor.fetchone()[0]
    conn.commit()

    cursor.close()
    conn.close()