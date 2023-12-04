#main file where flask initialize

from flask import Flask, jsonify
from content_based_filtering_maths import get_recommendations
from collaborative_filtering_math_flaskVersion import generate_recommendation
# import requests
# import json
import pandas as pd
import random
import os
import mysql.connector

app = Flask(__name__)

#SQL connection for Docker
# mydb = mysql.connector.connect(
#     host="mysql",
#     user="root",
#     password="khcy6ycy",
#     database="recommendation_engine",
#     port=3306
# )
mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="khcy6ycy",
    database="recommendation_engine",
    port=3306
)
print("Database connected")

# @app.route('/landingPage')
# def recommendedItem():
#     results = startRecommend()
#     return jsonify({"payload" : results})

# @app.route('/graphql')
# def get_graphql_data():
#     # sourcery skip: remove-unnecessary-else, swap-if-else-branches, use-fstring-for-formatting
#     # Here we define our query as a multi-line string
#     query = '''
#     query ($id: Int) { # Define which variables will be used in the query (id)
#     Media (id: $id, type: ANIME) { # Insert our variables into the query arguments (id) (type: ANIME is hard-coded in the query)
#         id
#         title {
#         romaji
#         english
#         native
#         }
#     }
#     }
#     '''

#     # Define our query variables and values that will be used in the query request
#     variables = {
#         'id': 15125
#     }

#     url = 'https://graphql.anilist.co'

#     # Make the HTTP Api request
#     response = requests.post(url, json={'query': query, 'variables': variables})
#     dataReturned = str(jsonify(response.json()))
    
#     if response.status_code == 200:
#         return jsonify(response.json())

#     else:
#         return f'Error : {str(response.status_code)}'


# @app.route('/analyzeData')
# def analyzeData():
#     query = '''
#     query ($id: Int) { # Define which variables will be used in the query (id)
#     Media (id: $id, type: ANIME) { # Insert our variables into the query arguments (id) (type: ANIME is hard-coded in the query)
#         id
#         title {
#         romaji
#         english
#         native
#         }
#     }
#     }
#     '''

#     # Define our query variables and values that will be used in the query request
#     variables = {
#         'id': 15125
#     }

#     url = 'https://graphql.anilist.co'

#     # Make the HTTP Api request
#     response = requests.post(url, json={'query': query, 'variables': variables})

#     #obtain dataReturned and pass back to recommendation engine for string manipulation purposes for POC
#     dataReturned = json.dumps(response.json())
#     print(f'Data returned : {dataReturned}')
#     return (modify(dataReturned))


#Define 2 API for math resources recommendation

#Recommendation based on Content-based Filtering for tackling cold start problems
@app.route('/contentBasedRecommendationMath')
def mathContentBasedFiltering():
    
    #for testing purpose to generate dynamic randomization
    sql_query = "SELECT * FROM recommendation_engine.mathresources"
    df = pd.read_sql(sql=sql_query, con=mydb)
    rsc_title = df['title'].values
    rand_max_limit = len(rsc_title)
    rand_rsc_title =  random.randint(0,rand_max_limit-1)
    title  = rsc_title[rand_rsc_title]
    recommended_list = get_recommendations(title).values
    #return recommendations list in the form of JSON to be used by Front-End
    return jsonify({
        "title" : title,
        "payload" : recommended_list.tolist()
        })
    

#Recommendation based on Collaborative-Filtering using Surprise Library for user-based recommendation
@app.route('/collaborativeFilteringMath')
def mathCollaborativeFiltering():

    user_id = random.randint(1,600)
    n_items = 10

    recommended_list = generate_recommendation(user_id, n_items)
    return jsonify(
        {
            "payload": recommended_list,
            "msg": f"Top {str(n_items)} Recommendation for user : {str(user_id)}",
        }
    )

@app.route('/')
def landingPage():
    return "Hello World!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
