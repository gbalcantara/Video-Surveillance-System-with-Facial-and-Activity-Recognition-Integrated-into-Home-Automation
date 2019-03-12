#!/usr/bin/python

import psycopg2

conn = psycopg2.connect(database="dateandtime", user = "gab", password = "admin1234", host = "127.0.0.1", port = "5432")

print ("Opened database successfully")

cur = conn.cursor()

#cur.execute("UPDATE COMPANY set SALARY = 25000.00 where ID = 1")
#conn.commit()
#print ("Total number of rows updated :", cur.rowcount)

cur.execute("SELECT id, fn, mdytm, zn, ps, ac from db11 ORDER BY id")
rows = cur.fetchall()
id1 = cur.rowcount
print (id1)

for row in rows:
   print ("ID = ", row[0])
   print ("FN = ", row[1])
   print ("MDYTM = ", row[2])
   print ("ZN = ", row[3])
   print ("PS = ", row[4])
   print ("AC = ", row[5], "\n")

print ("Operation done successfully")
conn.close()
conn.close()
