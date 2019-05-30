#!/usr/bin/python

import psycopg2

conn = psycopg2.connect(database="dateandtime", user = "gab", password = "admin1234", host = "127.0.0.1", port = "5432")

print ("Opened database successfully")

cur = conn.cursor()
cur.execute('''CREATE TABLE db111
      (ID INT PRIMARY KEY     NOT NULL,
      FN           TEXT,
      MDYTM        TIMESTAMP,
      ZN        TEXT,
      MD        BOOLEAN,
      PS        TEXT,
      AC        TEXT
      );''')
print ("Table created successfully")

conn.commit()
conn.close()


