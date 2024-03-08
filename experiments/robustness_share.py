import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gplearn.gplearn.genetic import SymbolicRegressor, SymbolicClassifier
from gplearn.gplearn.model import ShapeNN
from experiments.utils import create_df_from_cached_results, load_share_from_checkpoint, evaluate_shape, extract_slope_intercept
import torch
from sklearn.metrics import r2_score
from experiments.benchmarks_2 import run_experiment, categorical_variables_per_dataset, create_categorical_variable_dict
from experiments.load_data import load_data
from datetime import datetime
import json
from copy import deepcopy
import argparse

from temperature import generate_data

def test_share(device, n_jobs, population_size, generations, seed):
    task = 'regression'
    global_seed = 42

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
        'random_state':seed,
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
            'weight_decay': 1e-4,
            'alg':'adam',
            'lr': 1e-2, # tuned automatically
            'max_n_epochs':200,
            'tol':1e-3,
            'task':task,
            'device':device,
            'batch_size':64,
            'shape_class':ShapeNN,
            'constructor_dict': constructor_dict_ShapeNN,
            'num_workers_dataloader': 0,
            'seed':42,
            'checkpoint_folder': 'checkpoints',
            'enable_progress_bar': False,
            'patience': 25,
            'check_val_every_n_epoch': 1,
            }
        }

    esr = SymbolicRegressor(**gp_config, categorical_variables={})
    esr.fit(X,y)
    return esr

if __name__ == '__main__':

    # Argparse get a list of noise levels
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_levels', nargs='+', type=float)
    args = parser.parse_args()
    noise_levels = args.noise_levels

    global_seed = 42

    # Generate seeds for 3 runs using a generator
    gen = np.random.default_rng(global_seed)
    all_random_seeds = gen.choice(10000, 1000, replace=False)
    random_seeds = all_random_seeds[:3]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df_results = pd.DataFrame(columns=['run','seed','noise','r2'])

    for run, seed in enumerate(random_seeds):

       
        for noise in noise_levels:
            
            # Generate data
            df = generate_data(1000, (1,800), (1.0,4.0), (-100,0), seed=0, noise=noise)

            feature_columns = ['energy','mass','initial_temp']
            target_column = 'temperature'

            X = df[feature_columns].values
            y = df[target_column].values

            df_test = generate_data(1000, (1,800), (1.0,4.0), (-100,0), seed=1, noise=noise)
            X_test = df_test[feature_columns].values
            y_test = df_test[target_column].values

            # Run experiment
            device = 'cuda'
            n_jobs = 1
            population_size = 100
            generations = 10

            esr = test_share(device, n_jobs, population_size, generations, seed)
            y_pred = esr.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            # Add result to df using concat
            df_results = pd.concat([df_results, pd.DataFrame({'run':[run],'seed':[seed],'noise':[noise], 'r2':[r2]})], ignore_index=True)

            # Save df
            filepath = f'results/robustness/share_{timestamp}.csv'
            
            # Check if folder exists and create if not
            import os
            if not os.path.exists('results/robustness'):
                os.makedirs('results/robustness')
            
            df_results.to_csv(filepath)


        






