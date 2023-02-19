#main file where flask initialize

from flask import Flask, jsonify
from content_based_filtering_recommender import startRecommend, modify
import requests
import json

app = Flask(__name__)

@app.route('/landingPage')
def recommendedItem():
    results = startRecommend()
    return jsonify({"payload" : results})

@app.route('/graphql')
def get_graphql_data():
    # sourcery skip: remove-unnecessary-else, swap-if-else-branches, use-fstring-for-formatting
    # Here we define our query as a multi-line string
    query = '''
    query ($id: Int) { # Define which variables will be used in the query (id)
    Media (id: $id, type: ANIME) { # Insert our variables into the query arguments (id) (type: ANIME is hard-coded in the query)
        id
        title {
        romaji
        english
        native
        }
    }
    }
    '''

    # Define our query variables and values that will be used in the query request
    variables = {
        'id': 15125
    }

    url = 'https://graphql.anilist.co'

    # Make the HTTP Api request
    response = requests.post(url, json={'query': query, 'variables': variables})
    dataReturned = str(jsonify(response.json()))
    
    if response.status_code == 200:
        return jsonify(response.json())

    else:
        return f'Error : {str(response.status_code)}'


@app.route('/analyzeData')
def analyzeData():
    query = '''
    query ($id: Int) { # Define which variables will be used in the query (id)
    Media (id: $id, type: ANIME) { # Insert our variables into the query arguments (id) (type: ANIME is hard-coded in the query)
        id
        title {
        romaji
        english
        native
        }
    }
    }
    '''

    # Define our query variables and values that will be used in the query request
    variables = {
        'id': 15125
    }

    url = 'https://graphql.anilist.co'

    # Make the HTTP Api request
    response = requests.post(url, json={'query': query, 'variables': variables})

    #obtain dataReturned and pass back to recommendation engine for string manipulation purposes for POC
    dataReturned = json.dumps(response.json())
    print(f'Data returned : {dataReturned}')
    return (modify(dataReturned))


if __name__ == '__main__':
    app.run()
