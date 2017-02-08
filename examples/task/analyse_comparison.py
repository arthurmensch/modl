from datetime import datetime
from os.path import expanduser, join

from pymongo import MongoClient
import pandas as pd

client = MongoClient()
db = client['amensch']
runs = db['runs']


def first_trial():
    # Build dataframe
    results = runs.aggregate([{"$match": {'experiment.name': 'task_predict',
                               '_id': {"$gte": 184, "$lt": 195}}},
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
    df.to_csv(join(expanduser('~/output'), 'compare_task_predict_1.csv'),
              index=False)


def second_trial():
    results = runs.aggregate([{"$match": {'experiment.name': 'task_predict',
                               'start_time': {
                                   "$gte": datetime(2017, 2, 6, 12, 57),
                                   "$lt": datetime(2017, 2, 6, 13, 00)}}
                               },
                              {"$sort": {'_id': -1}},
                              {"$skip": 0},
                              {"$limit": 18},
                              {"$project": {
                                  'alpha': '$config.decomposition.alpha',
                                  'n_components': '$config.'
                                                  'decomposition.n_components',
                                  'train_size': '$config.rest_data.train_size',
                                  'test_score': '$info.pred_test_score',
                                  'train_score': '$info.pred_train_score',
                                  'unsupervised_score':
                                      '$info.unsupervised_score',
                                  'best_C': '$info.pred_best_C'
                              }}])
    df = pd.DataFrame(list(results))
    df.to_csv(join(expanduser('~/output'), 'compare_task_predict_2.csv'),
              index=False)

def third_trial():
    # Build dataframe
    results = runs.aggregate([{"$match": {'experiment.name': 'task_predict',
                               '_id': {"$gte": 284, "$lte": 301}}},
                              {"$sort": {'_id': -1}},
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
    df.to_csv(join(expanduser('~/output'), 'compare_task_predict_3.csv'),
              index=False)

if __name__ == '__main__':
    third_trial()