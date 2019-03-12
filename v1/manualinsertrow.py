#!/usr/bin/python

import psycopg2

conn = psycopg2.connect(database="dateandtime", user = "gab", password = "admin1234", host = "127.0.0.1", port = "5432")

print ("Opened database successfully")

cur = conn.cursor()

cur.execute("SELECT id, fn, mdytm, zn, ps, ac from db11")

id1 = cur.rowcount
id2 = id1 + 1
id2 = str(id2)
#print(id2)

cur.execute("INSERT INTO db11 (ID,MDYTM) VALUES (%s,current_timestamp)", (id2))

conn.commit()
print ("Records created successfully")
conn.close()
