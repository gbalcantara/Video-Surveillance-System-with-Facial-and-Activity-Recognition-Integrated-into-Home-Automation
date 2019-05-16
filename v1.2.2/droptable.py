#!/usr/bin/python

import psycopg2
import datetime

currentDT = datetime.datetime.now()
aa = currentDT.strftime("%m/%d/%Y")
bb = currentDT.strftime("%a")
cc = currentDT.strftime("%H:%M:%S")

#print (MDY1)
#print (DY1)
#print (TM1)

conn = psycopg2.connect(database="dateandtime", user = "gab", password = "admin1234", host = "127.0.0.1", port = "5432")
conn.autocommit = True
print ("Opened database successfully")

cur = conn.cursor()

cur.execute("DROP TABLE db2")

conn.commit()
print ("Records created successfully")
conn.close()
