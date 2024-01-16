#Creating and training models

from pyBKT.models import Model
import pandas as pd

def BKT():
    model = Model(seed=42, num_fits=1)
    # df = pd.read_csv('test.csv')
    # print(df.to_string()) 
    # print(model.params())
    test_df = pd.read_csv('VlearnedFlaskWithGraphQL/KTModels/data/test.csv')
    defaults = {'user_id': 'studentID', 'skill_name': 'skillComponent', 'correct': 'Correct', 'start_time': 'start',
                'end_time': 'end', 'multigs': 'studentID'}
    skill = ["addition", "subtraction"]
    model.fit(data = test_df, defaults=defaults, skills=skill, multilearn='studentID', multigs=True)
    training_rmse = model.evaluate(data = test_df)
    training_auc = model.evaluate(data = test_df, metric = 'auc')
    print("Training RMSE: %f" % training_rmse)
    print("Training AUC: %f" % training_auc)
    print("Model Parameters : \n")
    print(model.params())

    #predict data
    preds = model.predict(data=test_df)
    print(preds.head(5))

if __name__ == "__main__":
    BKT()