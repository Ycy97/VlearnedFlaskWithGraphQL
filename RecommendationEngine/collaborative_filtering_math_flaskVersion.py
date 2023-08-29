#https://medium.com/tiket-com/get-to-know-with-surprise-2281dd227c3e

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split

def generate_recommendation(user_id, n_items):

    ratings_data = pd.read_csv("resource_ratings.csv", low_memory=False)
    resource_data = pd.read_csv("mathResources.csv", low_memory=False)

    min_rating = ratings_data.rating.min()
    max_rating = ratings_data.rating.max()

    reader = Reader(rating_scale=(min_rating, max_rating))
    data = Dataset.load_from_df(ratings_data[['userId', 'resourceId', 'rating']], reader)

    svd = SVD(n_epochs=10)

    param_grid = {
        'n_factors': [20, 50, 100],
        'n_epochs' : [5, 10, 20]
    }

    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=10)
    gs.fit(data)
    best_factor = gs.best_params['rmse']['n_factors']
    best_epoch = gs.best_params['rmse']['n_epochs']
    trainset, testset = train_test_split(data, test_size=.20)
    svd = SVD(n_factors=best_factor, n_epochs=best_epoch)
    svd.fit(trainset)

    resource_ids  = ratings_data["resourceId"].unique()

    #get a list of all resources IDs that have been read by the users
    resource_ids_user = ratings_data.loc[ratings_data["userId"] == user_id, "resourceId"]

    #get a list of all resources Ids that have not been read by the users
    resource_ids_to_pred = np.setdiff1d(resource_ids, resource_ids_user)

    #apply a rating of 4 to all interactions (only to match the Surprise dataset format)
    test_set = [[user_id, resource_ids, 4] for resource_ids in resource_ids_to_pred]

    #predict the ratings and generate recommendations
    predictions = svd.test(test_set)
    pred_ratings = np.array([pred.est for pred in predictions])
    print("Top {0} item recommendations for user {1} :".format(n_items,user_id))

    #Rank top-N resources based on the predicted ratings
    index_max = (pred_ratings).argsort()[:10]

    final_list = []
    for i in index_max:
        resource_ids = resource_ids_to_pred[i]
        #print(resource_data[resource_data["resource_id"] == resource_ids]["title"].values[0])
        final_list.append(resource_data[resource_data["resource_id"] == resource_ids]["title"].values[0])

    return final_list