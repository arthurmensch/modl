from os.path import expanduser, join

from pymongo import MongoClient
import pandas as pd

client = MongoClient()
db = client['amensch']
runs = db['runs']

# Build dataframe
results = runs.aggregate([{"$match": {'experiment.name': 'task_predict'}},
                          {"$sort": {'_id': -1}},
                          {"$skip": 1},
                          {"$limit": 18},
                          {"$project": {
                              'alpha': '$config.decomposition.alpha',
                              'n_components': '$config.decomposition.n_components',
                              'train_size': '$config.rest_data.train_size',
                              'test_score': '$info.pred_test_score',
                              'train_score': '$info.pred_train_score',
                              'dec_score': '$info.dec_final_score',
                              'best_C': '$info.best_C'
                          }}])
df = pd.DataFrame(list(results))
df.to_csv(join(expanduser('~/output'), 'compare_task_predict.csv'),
          index=False)
