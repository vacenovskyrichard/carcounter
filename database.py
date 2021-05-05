import sqlite3

conn = sqlite3.connect('database.db')

cursor = conn.cursor()


# cursor.execute("""CREATE TABLE vehicles(
#                   video text,
#                   counter integer
#                   )""")
#
cursor.execute("SELECT * FROM vehicles WHERE video='vertical.mp4'")
print(cursor.fetchall())

conn.commit()
conn.close()

