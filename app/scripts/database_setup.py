import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

con = psycopg2.connect(
    dbname='postgres',
    user='postgres',
    host='localhost',
    password='postgres'
)

con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

cur = con.cursor()
print(cur.name)

cur.execute(sql.SQL("CREATE DATABASE {}").format(
        sql.Identifier('stadprin'))
)

cur.close()

# https://www.datacamp.com/tutorial/tutorial-postgresql-python
# The above script will create a new database called stadprin. 
# Now write Python code that will create the following tables:
# `visual_analysis` - with columns: ID cards detected, Scene sentiment, People detected, Cars detected, Cars detected, Objects detected, Frame activity, batch_number.
# `audio_analysis` - with columns: summary, sentiment, tone, fixed_transcript, urgent.
