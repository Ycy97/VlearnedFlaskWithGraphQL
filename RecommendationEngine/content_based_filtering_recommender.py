#Recommendation Engine using Content-based Filtering

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#Reading data from CSV file, will be replacing it with GraphQL API call to retrieve data and process into matrix
df = pd.read_csv("VlearnedFlaskWithGraphQL\RecommendationEngine\mathResources.csv")

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=0, stop_words='english')

#to use genre column to compute cosine similarities
tfidf_matrix = tf.fit_transform(df['genre'])
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
results = {}

#for loop to compute cosine similarities in each row of the matrix
for idx, row in df.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
    similar_items = [(cosine_similarities[idx][i], df['resource_id'][i]) for i in similar_indices]

    results[row['resource_id']] = similar_items[1:]

print('Cosine Similarities Computation Completed!')

def item(id):
    return df.loc[df['resource_id'] == id]['genre'].tolist()[0].split(' - ')[0]

def recommend(item_id, num):  # sourcery skip: for-append-to-extend, list-comprehension
    #print(f"Recommending {str(num)} products similar to " + item(item_id) + "....")
    #print("----------")
    recs = results[item_id][:num]

    itemList = []

    #for loop to print out all the results
    for rec in recs:
        #print("Recommended item: " + item(rec[1]))
        itemList.append(str(item(rec[1])))
    
    print(itemList)
    return itemList


def startRecommend():
    return recommend(item_id=6, num=5)


#POC method for string manipulation
def modify(graphQLResponse):
    return "Modified : " + graphQLResponse
    
startRecommend()