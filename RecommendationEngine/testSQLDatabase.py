#Connecting to SQL database using mysql.connector

import mysql.connector
import pandas as pd

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="khcy6ycy",
    database="recommendation_engine"
)

mycursor = mydb.cursor()

sql_query = "SELECT * FROM mathresources"

mycursor.execute(sql_query)

df_sql_data = pd.DataFrame(mycursor.fetchall())

print(df_sql_data)

sql_query2 = "SELECT * FROM resource_ratings"

mycursor.execute(sql_query2)

df_sql_data2 = pd.DataFrame(mycursor.fetchall())

print(df_sql_data2)

