#Backup for Creating and training BKT models using pyBKT

from pyBKT.models import Model
import pandas as pd

def BKT():
    model = Model(seed=42, num_fits=1)
    # df = pd.read_csv('test.csv')
    # print(df.to_string()) 
    # print(model.params())
    test_df = pd.read_csv('VlearnedFlaskWithGraphQL/KTModels/data/test.csv')
    test2_df = pd.read_csv('VlearnedFlaskWithGraphQL/KTModels/data/test2.csv')
    defaults = {'user_id': 'studentID', 'skill_name': 'skillComponent', 'correct': 'Correct', 'start_time': 'start',
                'end_time': 'end', 'multilearn': 'studentID', 'multigs' : True}
    # defaults = {'user_id': 'studentID', 'skill_name': 'skillComponent', 'correct': 'Correct', 'start_time': 'start',
    #             'end_time': 'end'}
    #skill = ["addition", "subtraction"]
    #model.fit(data = test2_df, defaults=defaults, skills=skill, multilearn='studentID', multigs=True)
    model.fit(data = test2_df, defaults=defaults)
    training_rmse = model.evaluate(data = test2_df)
    training_auc = model.evaluate(data = test2_df, metric = 'auc')
    print("Training RMSE: %f" % training_rmse)
    print("Training AUC: %f" % training_auc)
    print("Model Parameters : \n")
    print(model.params())

    #predict data
    preds = model.predict(data=test2_df)
    print(preds.head(10))

if __name__ == "__main__":
    BKT()