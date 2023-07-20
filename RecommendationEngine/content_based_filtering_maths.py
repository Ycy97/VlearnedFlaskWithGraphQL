#Content-Based Recommender for Maths Resources(self-generated)

import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#Reading CSV file
df = pd.read_csv("VlearnedFlaskWithGraphQL\RecommendationEngine\mathResources.csv", low_memory=False)
#print(df)

tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(df['genre'])
#print(tfidf_matrix.shape)
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title']).drop_duplicates()
#print(indices[:10])

def get_recommendations(title, consine_sim=cosine_similarities):
    #Get the index of the movie that matches the title
    idx = indices[title]
    #print("The index of " + title + " is: " + str(idx))

    #get the pairwise similarity scroes of all resources with targeted resource
    sim_scores = list(enumerate(consine_sim[idx]))
    #print(sim_scores)

    # Sort the resource based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    #Get the scores of the 10 most similar resource
    sim_scores = sim_scores[1:11]

    #Get the resource indices
    rsc_indices  = [i[0] for i in sim_scores]
    #print(rsc_indices)

    return df['title'].iloc[rsc_indices]

#test data purposes using randomizer, main file will be calling this function instead
rsc_title = df['title'].values
#print(rsc_title)
#print(type(rsc_title))
rand_max_limit = len(rsc_title)
#print(f"Number of elements in array : {rand_max_limit}")

#creates randomizer for recommendation
rand_rsc_title =  random.randint(0,rand_max_limit)
print(rand_rsc_title)

print("Top 10 Recommended Resource you may like based on " + rsc_title[rand_rsc_title] + "\n")
recommended_list = get_recommendations(rsc_title[rand_rsc_title]).values

#for loop to list out the resources title
for x in recommended_list:
    print(x)