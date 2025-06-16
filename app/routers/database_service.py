from datetime import datetime
import os
import psycopg2
import json

# Dictionary with DB_CONFIG
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "database": os.getenv("POSTGRES_DB", "stadprin"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "posty"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
}


# Connect to "PostgreSQL" database
def connect_to_db():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print(
            f"Database connection established to {DB_CONFIG['database']} at {DB_CONFIG['host']}"
        )
        return conn, cursor
    except Exception as e:
        print(f"Failed to connect to database: {e}")
        return None, None


class Db_helper:
    def __init__(self):
        self.db_config = DB_CONFIG
        self.conn = None
        self.cursor = None

    # Document Process data

    def video_analysis(self, analysis_data, source_path=None):
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()

            # Extract data from analysis_data
            frame_description = analysis_data.get("Frame description", "")
            objects_detected = analysis_data.get("Objects detected", [])
            objects_detected = ", ".join(objects_detected)

            cars_detected = analysis_data.get("License plates", [])
            people_detected = analysis_data.get("People detected", [])

            scene_sentiment = analysis_data.get("Scene sentiment", "")
            id_cards_detected = analysis_data.get("ID cards detected", [])

            insert_sql = """
            INSERT INTO visual_analysis
            (frame_activity, objects_detected, cars_detected, people_detected, 
            scene_sentiment, id_cards_detected, batch_number, source_path, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
            """

            values = (
                frame_description,
                objects_detected,
                cars_detected,
                people_detected,
                scene_sentiment,
                id_cards_detected,
                datetime.now(),            )
            print(values)
            self.cursor.execute(insert_sql, values)
            analysis_id = self.cursor.fetchone()[0]
            
            self.conn.commit()
            return analysis_id
        except Exception as e:
            print(f"Database error: {e}")
            if self.conn:
                self.conn.rollback()
            return None
        finally:            # Always close connections
            if self.cursor:
                self.cursor.close()
            if self.conn and not self.conn.closed:
                self.conn.close()
    
    # def store_document(self, document_name, document_path, field_names=None, video_processing_prompt=None,
    #                   video_path=None, source_path=None, document_schema=None, generated_system_prompt=None,
    #                   client_id="default", document_title=None, document_type="form", created_by="system"):
    #     """
    #     Store document with enhanced schema support for dynamic prompt generation.
        
    #     Args:
    #         document_name: Name of the document
    #         document_path: Path to the document file
    #         field_names: Legacy field names (for backward compatibility)
    #         video_processing_prompt: Video processing configuration
    #         video_path: Path to associated video file
    #         source_path: Source path of the document
    #         document_schema: DocumentSchema object with enhanced field information
    #         generated_system_prompt: Generated system prompt text
    #         client_id: Client identifier
    #         document_title: Title of the document
    #         document_type: Type of document (form, medical, legal, etc.)
    #         created_by: User/system that created the document
    #     """
    #     try:
    #         import json
    #         self.conn = psycopg2.connect(**self.db_config)
    #         self.cursor = self.conn.cursor()

    #         # Use document_title if provided, otherwise use document_name
    #         final_document_title = document_title or document_name

    #         # Prepare schema_fields from document_schema if provided
    #         schema_fields_json = None
    #         if document_schema:
    #             if hasattr(document_schema, 'schema_fields'):
    #                 # Convert DocumentSchema object to dict format suitable for JSONB
    #                 schema_fields_dict = {}
    #                 for field_name, schema_field in document_schema.schema_fields.items():
    #                     if hasattr(schema_field, 'model_dump'):
    #                         schema_fields_dict[field_name] = schema_field.model_dump()
    #                     else:
    #                         # Fallback for dict-like objects
    #                         schema_fields_dict[field_name] = dict(schema_field)
    #                 schema_fields_json = json.dumps(schema_fields_dict)
    #             elif isinstance(document_schema, dict):
    #                 # Handle dict format
    #                 schema_fields_json = json.dumps(document_schema.get('schema_fields', {}))
            
    #         # Fallback: create basic schema from field_names for backward compatibility
    #         if not schema_fields_json and field_names:
    #             schema_fields_dict = {}
    #             field_list = field_names if isinstance(field_names, list) else [field_names]
    #             for field_name in field_list:
    #                 schema_fields_dict[field_name] = {
    #                     "label": field_name,
    #                     "field_type": "text",
    #                     "required": True,
    #                     "description": ""
    #                 }
    #             schema_fields_json = json.dumps(schema_fields_dict)

    #         insert_sql = """
    #         INSERT INTO form_documents
    #         (client_id, document_title, document_type, is_active, schema_fields, 
    #          generated_system_prompt, video_processing_prompt, document_path, 
    #          video_path, source_path, created_by, created_on, processed_date)
    #         VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s, %s, %s, %s, %s, %s)
    #         RETURNING id;
    #         """

    #         current_time = datetime.now()
    #           # Convert video processing prompt to JSON string for postgres
    #         video_prompt_string = None
    #         if video_processing_prompt:
    #             if isinstance(video_processing_prompt, (dict, list)):
    #                 video_prompt_string = json.dumps(video_processing_prompt)
    #             else:
    #                 try:
    #                     parsed = json.loads(video_processing_prompt)
    #                     video_prompt_string = json.dumps(parsed)
    #                 except (json.JSONDecodeError, TypeError):
    #                     video_prompt_string = json.dumps({"prompt": str(video_processing_prompt)})
            
    #         values = (
    #             client_id,
    #             final_document_title,
    #             document_type,
    #             True,  # is_active
    #             schema_fields_json,  # JSONB
    #             generated_system_prompt,  # TEXT
    #             video_prompt_string,  # JSONB
    #             document_path,
    #             video_path,
    #             source_path,
    #             created_by,
    #             current_time,  # created_on
    #             current_time   # processed_date
    #         )

    #         self.cursor.execute(insert_sql, values)
    #         document_id = self.cursor.fetchone()[0]

    #         self.conn.commit()
    #         print(f"Stored document '{final_document_title}' with ID: {document_id}")
            
    #         return document_id

    #     except Exception as e:
    #         print(f"Database error in store_document: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         if self.conn:
    #             self.conn.rollback()
    #         return None

    #     finally:
    #         if self.cursor:
    #             self.cursor.close()
    #         if self.conn and not self.conn.closed:
    #             self.conn.close()

    def audio_analysis(self, analysis_data, source_path=None):
        try:
            pass
            # self.conn = psycopg2.connect(**self.db_config)
            # self.cursor = self.conn.cursor()

            # # Extract data from analysis_data
            # frame_description = analysis_data.get('Frame description', '')
            # objects_detected = analysis_data.get('Objects dectected', [])
            # objects_detected = ", ".join(objects_detected)

            # license_plates = analysis_data.get('License plates', [])
            # license_plates = ", ".join(license_plates)

            # scene_sentiment = analysis_data.get('Scene sentiment', '')
            # risk_analysis = analysis_data.get('Risk analysis', '')

            # insert_sql = """
            # INSERT INTO visual_analysis
            # (frame_description, objects_detected, license_plates, scene_sentiment,
            # risk_analysis, created_at)
            # VALUES (%s, %s, %s, %s, %s, %s)
            # RETURNING id;
            # """

            # values = (
            #     frame_description,
            #     objects_detected,
            #     scene_sentiment,
            #     license_plates,
            #     risk_analysis,
            #     datetime.now()
            # )
            # print(frame_description, objects_detected, license_plates, scene_sentiment,
            # risk_analysis, datetime.now())

            # self.cursor.execute(insert_sql, values)
            # analysis_id = self.cursor.fetchone()[0]

            # self.conn.commit()
            # return analysis_id

        except Exception as e:
            print(f"Database error: {e}")
            if self.conn:
                self.conn.rollback()
            return None

        finally:
            # Always close connections
            if self.cursor:
                self.cursor.close()
            if self.conn and not self.conn.closed:
                self.conn.close()

    def get_session(self):
        """
        Create and return a SQLAlchemy session
        """
        try:
            from sqlalchemy import create_engine
            from sqlalchemy.orm import sessionmaker
            
            # Create connection string
            conn_str = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"

            # Create engine
            engine = create_engine(conn_str)

            # Create session
            Session = sessionmaker(bind=engine)
            return Session()

        except Exception as e:
            print(f"Error creating session: {e}")
            return None
