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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str, default='servo', help='Name of the dataset')
    parser.add_argument('task', type=str, default='regression', help='Task type (regression or classification)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (default: cuda)')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of jobs (default: 1)')
    parser.add_argument('--population_size', type=int, default=100, help='Population size (default: 100)')
    parser.add_argument('--generations', type=int, default=10, help='Number of generations (default: 10)')
    parser.add_argument('--global_seed', type=int, default=42, help='Global seed (default: 42)')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials (default: 10)')
    parser.add_argument('--start_from_trial', type=int, default=0, help='Start from trial (default: 0)')

    args = parser.parse_args()

    dataset_name = args.dataset_name
    task = args.task
    device = args.device
    n_jobs = args.n_jobs
    population_size = args.population_size
    generations = args.generations
    global_seed = args.global_seed
    n_trials = args.n_trials
    start_from_trial = args.start_from_trial

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
            'weight_decay': 1e-4,
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

    esr_parameter_dict = None
    if task == 'regression':
        esr = SymbolicRegressor(**gp_config, categorical_variables=create_categorical_variable_dict(dataset_name,task))
    elif task == 'classification':
        esr = SymbolicClassifier(**gp_config, categorical_variables=create_categorical_variable_dict(dataset_name,task))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    gp_config_to_save = deepcopy(gp_config)
    gp_config_to_save['optim_dict'].pop('shape_class')

    metadata = {
        'dataset_name':dataset_name,
        'task':task,
        'timestamp':timestamp,
        'gp_config':gp_config_to_save,
        'global_seed':global_seed,
        'device':device,
        'n_jobs':n_jobs,
        'n_trials':n_trials,
        'start_from_trial':start_from_trial
    }

    # Save metadata
    with open(f"results/{timestamp}_metadata.json", "w") as outfile:
        json.dump(metadata, outfile, indent=4)

    esr_score_mean, esr_score_std, model = run_experiment(dataset_name, esr, esr_parameter_dict, task, random_state=global_seed, return_model=True, timestamp=timestamp, n_trials=n_trials, start_from_trial=start_from_trial)
    print(esr_score_std)

    df = create_df_from_cached_results(model.cached_results)

    df.to_csv(f"{model.timestamp}_cached_results.csv")

