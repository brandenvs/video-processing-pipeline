import sqlite3

def write_to_db(formatted_original: str, analysis:dict):
    # Connect to SQLite database
    conn = sqlite3.connect("temp.db")
    cursor = conn.cursor()

    # Create table structure
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcript_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_transcript TEXT,
            fixed_transcript TEXT,
            summary TEXT,
            sentiment_label TEXT,
            sentiment_score REAL,
            urgent INTEGER
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tone_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_id INTEGER,
            label TEXT,
            score REAL,
            FOREIGN KEY(transcript_id) REFERENCES transcript_analysis(id)
        )
    ''')

    # Insert into transcript_analysis table
    cursor.execute('''
        INSERT INTO transcript_analysis (
            original_transcript, fixed_transcript, summary,
            sentiment_label, sentiment_score, urgent
        ) VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        formatted_original,
        analysis["fixed_transcript"],
        analysis["summary"],
        analysis["sentiment"]["label"],
        analysis["sentiment"]["score"],
        int(analysis["urgent"])
    ))

    transcript_id = cursor.lastrowid

    # Insert into tone_analysis table
    tone_data = list(zip(analysis["tone"]["labels"], analysis["tone"]["scores"]))
    cursor.executemany('''
        INSERT INTO tone_analysis (transcript_id, label, score)
        VALUES (?, ?, ?)
    ''', [(transcript_id, label, score) for label, score in tone_data])

    # Commit and close
    conn.commit()
    conn.close()

