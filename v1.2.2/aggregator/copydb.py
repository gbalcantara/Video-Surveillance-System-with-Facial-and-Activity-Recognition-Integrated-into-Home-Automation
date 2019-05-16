import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_SERIALIZABLE
import sys
from io import StringIO

conn = psycopg2.connect(database="dateandtime", user = "gab", password = "admin1234", host = "192.168.0.104", port = "5432")
print("Opened database datandtime successfully")
cur = conn.cursor()

conn2 = psycopg2.connect(database="timeinterval", user = "sjeckem", password = "admin1234", host = "127.0.0.1", port = "5432")
print("Opened database timeinterval successfully")
cur2 = conn2.cursor()

# select * from db1 where id > 81168 order by id desc;
input = StringIO()
cur.copy_expert('COPY (select * from db11 WHERE id > 81168) TO STDOUT', input)
input.seek(0)
cur2.copy_expert('COPY db1 FROM STDOUT', input)
conn2.commit()
