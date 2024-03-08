# Import libraries
import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gplearn.gplearn.genetic import SymbolicRegressor
from gplearn.gplearn.model import ShapeNN
from experiments.temperature import generate_data
from experiments.utils import create_df_from_cached_results, load_share_from_checkpoint, get_n_shapes, get_n_variables
import torch
from sklearn.metrics import r2_score
from interpret.glassbox import ExplainableBoostingRegressor
from pysr import PySRRegressor
from interpret import show
import time
import argparse

if __name__ == '__main__':

    # Generate data
    df = generate_data(1000, (1,800), (1.0,4.0), (-100,0), seed=0, noise=0.0)

    feature_columns = ['energy','mass','initial_temp']
    target_column = 'temperature'

    X = df[feature_columns].values
    y = df[target_column].values

    df_test = generate_data(1000, (1,800), (1.0,4.0), (-100,0), seed=1, noise=0.0)
    X_test = df_test[feature_columns].values
    y_test = df_test[target_column].values

    def test_share(device, n_jobs, population_size, generations, batch_size):
        task = 'regression'
        global_seed = 1

        constructor_dict_ShapeNN = {
            'n_hidden_layers':5,
            'width':10,
            'activation_name':'ELU'
            }

        gp_config = {
            'population_size':population_size,
            'generations':generations,
            'tournament_size':10,
            'function_set':('add','mul','div','shape'),
            'verbose':True,
            'random_state':global_seed,
            'const_range':None,
            'n_jobs':n_jobs,
            'p_crossover':0.4,
            'p_subtree_mutation':0.2,
            'p_point_mutation':0.2,
            'p_hoist_mutation':0.05,
            'p_point_replace':0.2,
            'parsimony_coefficient':0.0,
            'metric': ('mse' if task == 'regression' else 'log loss'),
            'parsimony_coefficient':0.0,
            'optim_dict': {
                'alg':'adam',
                'lr': 1e-2, # tuned automatically
                'max_n_epochs':1000,
                'tol':1e-3,
                'task':task,
                'device':device,
                'batch_size':batch_size,
                'weight_decay': 1e-4,
                'shape_class':ShapeNN,
                'constructor_dict': constructor_dict_ShapeNN,
                'num_workers_dataloader': 0,
                'seed':42,
                'checkpoint_folder':'results/checkpoints',
                'keep_models':True
                }
            }

        esr = SymbolicRegressor(**gp_config, categorical_variables={})
        esr.fit(X,y)
        return esr

    t1 = time.time()
    population_size = 500
    generations = 10
    share = test_share('cpu',1,population_size,generations,2000)

    timestamp = share.timestamp

    # Load dictionary of results into a dataframe
    res_df = pd.read_csv(f'results/checkpoints/{timestamp}/dictionary.csv')

    # Add validation loss to the dataframe
    loss_val = []
    loss_train = []
    n_shapes = []
    n_variables = []
    for row in res_df[['id','equation']].itertuples(index=False):
        eq = row.equation
        id = row.id
        print(eq)
        n_shapes.append(get_n_shapes(eq))
        n_variables.append(get_n_variables(eq))
        esr = load_share_from_checkpoint(timestamp, eq, checkpoint_dir='results/checkpoints', task='regression',n_features=3, equation_id=id)
        loss_train.append(r2_score(y,esr.predict(X)))
        loss_val.append(r2_score(y_test,esr.predict(X_test)))
    res_df['r2_val'] = loss_val
    res_df['r2_train'] = loss_train
    res_df['n_shapes'] = n_shapes
    res_df['n_variables'] = n_variables

    res_df.to_csv(f'results/Figure_7_results.csv')

    t2 = time.time()

    # Save the time it took to run the experiment
    with open(f'results/Figure_7_time.txt','w') as f:
        f.write(f'{t2-t1}')

    # Save the timestamp
    with open(f'results/Figure_7_timestamp.txt','w') as f:
        f.write(timestamp)
    




