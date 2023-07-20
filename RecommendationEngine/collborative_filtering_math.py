import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
from surprise.model_selection import train_test_split


ratings_data = pd.read_csv("VlearnedFlaskWithGraphQL\RecommendationEngine\\resource_ratings.csv", low_memory=False)
resource_data = pd.read_csv("VlearnedFlaskWithGraphQL\RecommendationEngine\mathResources.csv", low_memory=False)

#print(ratings_data.head(10))
#print("---------------------------------------------------------------------------------------------------")
#print(resource_data.head(10))

#We need to convert our dataset into a Dataset object from the Surprise Library, we do this by defining a Reader Object
#to be able to parse the DataFrame

#Get min and max rating from dataset
min_rating = ratings_data.rating.min()
max_rating = ratings_data.rating.max()

print(min_rating)
print(max_rating)

reader = Reader(rating_scale=(min_rating, max_rating))
data = Dataset.load_from_df(ratings_data[['userId', 'resourceId', 'rating']], reader)

#We are using SVD, one of the most popular matrix factorization algorithm to predict the missing interaction on
#the user-item interaction matrix by performing factorization to produce user latent and item latent factors
svd = SVD(n_epochs=10)
results = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=10, verbose=True)

#print("Average MAE : ", np.average(results["test_mae"]))
#print("Average RMSE : ", np.average(results["test_rmse"]))

#Hyperparameter Tuning
#uses grid search cross-validation in hyperparameter tuning
param_grid = {
    'n_factors': [20, 50, 100],
    'n_epochs' : [5, 10, 20]
}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=10)
gs.fit(data)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])

#After the best hyperparameters are obtained, we can retrain the model using these hyperparameter
best_factor = gs.best_params['rmse']['n_factors']
best_epoch = gs.best_params['rmse']['n_epochs']

#sample random trainset and testset
#test set is made of 20% of the ratings
trainset, testset = train_test_split(data, test_size=.20)

#using the SVD algorithm
svd = SVD(n_factors=best_factor, n_epochs=best_epoch)

#train the algorithm on the trainset
svd.fit(trainset)

#Once we have trained the model with the best hyperparameters, we can provide recommendations to the users
# 1) find list of resources particular user has yet to seen
# 2) predict each interaction that is missing using the model
# 3) get top N resources recommendation by ranking them

def generate_recommendation(model, user_id, ratings_df, resource_df, n_items):
    #get a list of all resource IDs from dataset
    resource_ids  = ratings_df["resourceId"].unique()

    #get a list of all resources IDs that have been read by the users
    resource_ids_user = ratings_df.loc[ratings_df["userId"] == user_id, "resourceId"]

    #get a list of all resources Ids that have not been read by the users
    resource_ids_to_pred = np.setdiff1d(resource_ids, resource_ids_user)

    #apply a rating of 4 to all interactions (only to match the Surprise dataset format)
    test_set = [[user_id, resource_ids, 4] for resource_ids in resource_ids_to_pred]

    #predict the ratings and generate recommendations
    predictions = model.test(test_set)
    pred_ratings = np.array([pred.est for pred in predictions])
    print("Top {0} item recommendations for user {1} :".format(n_items,user_id))

    #Rank top-N resources based on the predicted ratings
    index_max = (pred_ratings).argsort()[:n_items]
    for i in index_max:
        resource_ids = resource_ids_to_pred[i]
        #print(resource_df[resource_df["resource_id"] == resource_ids]["title"].values[0], pred_ratings[i])
        print(resource_df[resource_df["resource_id"] == resource_ids]["title"].values[0])

    
#define which user ID to give recommendation
userID = 23
n_items = 10
generate_recommendation(svd,userID,ratings_data,resource_data,n_items)
