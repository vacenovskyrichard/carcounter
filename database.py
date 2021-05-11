import sqlite3

conn = sqlite3.connect('database.db')

cursor = conn.cursor()


# cursor.execute("""CREATE TABLE vehicles(
#                   video text,
#                   car_count integer,
#                   bus_count integer,
#                   truck_count integer,
#                   minute_1_count integer,
#                   minute_2_count integer,
#                   minute_3_count integer,
#                   minute_4_count integer,
#                   minute_5_count integer,
#                   total_count integer
#                   )""")



cursor.execute("SELECT * FROM vehicles WHERE video='test_1.mp4'")
print(cursor.fetchall())

conn.commit()
conn.close()

