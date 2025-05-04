from datetime import datetime
import os
import psycopg2

# Dictionary with DB_CONFIG
DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "database": os.getenv("POSTGRES_DB", "stadprin"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
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
                datetime.now(),
            )
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

        finally:
            # Always close connections
            if self.cursor:
                self.cursor.close()
            if self.conn and not self.conn.closed:
                self.conn.close()

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
