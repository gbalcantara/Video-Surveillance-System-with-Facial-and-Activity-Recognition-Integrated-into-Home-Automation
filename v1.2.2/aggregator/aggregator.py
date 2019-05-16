import psycopg2
from datetime import datetime, timedelta, time
import requests
import json

currentDT = datetime.now()
print(currentDT)

date = currentDT.date()
print(date)

yesterday = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')
#print(yesterday)

#date = 
date = datetime.strptime(yesterday,'%Y-%m-%d')

start_time = datetime.combine(date, time(7, 0, 0))
print(start_time)
end_time = start_time + timedelta(minutes=30) - timedelta(seconds=1)
print(end_time)
day = start_time.weekday()
print(day)

print(date)
time_interval = {
        "mty" : [],
        "mnill" : [],
        "jom" : [],
        "ntig" : [],
        "cdz" : []
        }

fac_id = {
    "mty" : 1,
    "mnill" : 2,
    "jom" : 3,
    "ntig" : 4,
    "cdz" : 5
    }

url = 'https://eyesmart.herokuapp.com/api/free-time/'
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

conn = psycopg2.connect(database="dateandtime", user = "gab", password = "admin1234", host = "127.0.0.1", port = "5432")

print("Opened database datandtime successfully")
cur = conn.cursor()

# loop for every time interval
for i in range(22):
    cur.execute("SELECT id, fn, mdytm, zn, ps, ac FROM db11 \
                WHERE ps IS NOT NULL AND mdytm BETWEEN '%s' AND '%s' ORDER BY id" \
                % (start_time, end_time))

    #print(start_time, end_time)
    start_time += timedelta(minutes=30)
    end_time += timedelta(minutes=30)
    # ID, FILENAME, DATE_TIME, ZONE, PERSON, ACTION

    rows = cur.fetchall()
    if not rows:
    #    print("No entries\n")
        continue

    count = {
            "mty" : 0,
            "mnill" : 0,
            "jom" : 0,
            "ntig" : 0,
            "cdz" : 0
            }

    init = rows[0]
    for index, row in enumerate(rows):
        id, _, curtime, _, ps, *_ = row
        prevtime = rows[index-1][2]
        #print("ID = ", id)
        #print("MDYTM = ", curtime)
        #print("PERSON = ", ps)
        #if row != init:
            #print(curtime - prevtime)
        #print("\n")
        if ps:
            count[ps] += 1

    #print("Total entries:")
    #print(count, "\n")

    #print("time_interval = ", i)
    #print("day = ", day)

    #print("\n")

    for ps, value in count.items():
        if value > 300:
            print(ps, value, start_time.date())
            time_interval[ps].append('%s' % i)

    print("\n")
print(time_interval)
for ps, intervals in time_interval.items():
    if intervals:
        conn = psycopg2.connect(database="timeinterval", user = "gab", password = "admin1234", host = "127.0.0.1", port = "5432")
        cur = conn.cursor()
        cur.execute("SELECT id, ps, date, day, time_interval FROM db2b")
        id1 = cur.rowcount
        id2 = id1 + 1
        cur.execute("INSERT INTO db2b (id, ps, date, day, time_interval) \
            VALUES (%s, '%s', '%s', %s, ARRAY %s)" % (id2, ps, date, day, intervals))
        conn.commit()
        print("Inserted data into database")

        data = {'faculty': '%s' % fac_id[ps],
                'free_time': intervals,
                'day': '%s' % day
                }
        print(json.dumps(data))
        r = requests.post(url, data=json.dumps(data), headers=headers)
        print(r)

conn.close()
conn.close()
print("Closed database successfully")
