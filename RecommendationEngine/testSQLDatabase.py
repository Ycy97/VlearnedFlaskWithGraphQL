#Connecting to SQL database using mysql.connector

import mysql.connector
import pandas as pd

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="khcy6ycy",
    database="recommendation_engine"
)

sql_query = "SELECT * FROM mathresources"

df_sql_data = pd.read_sql(sql=sql_query, con=mydb)

print(df_sql_data)

sql_query2 = "SELECT * FROM resource_ratings"

df_sql_data2 = pd.read_sql(sql=sql_query2, con=mydb)

print(df_sql_data2)

