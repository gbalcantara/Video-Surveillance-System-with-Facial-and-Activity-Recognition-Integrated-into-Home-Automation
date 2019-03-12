#!/usr/bin/python

import psycopg2

conn = psycopg2.connect(database="dateandtime", user = "gab", password = "admin1234", host = "127.0.0.1", port = "5432")

print ("Opened database successfully")

cur = conn.cursor()

cur.execute("DELETE FROM db11")
conn.commit()
#print ("Total number of rows deleted :", cur.rowcount)

#cur.execute("SELECT id, mdy, dy, tm  from DATE_TIME")
#rows = cur.fetchall()


#for row in rows:
#   print ("ID = ", row[0])
#   print ("MDY = ", row[1])
#   print ("DY = ", row[2])
#   print ("TM = ", row[3], "\n")

print ("Operation done successfully")
conn.close()
conn.close()
