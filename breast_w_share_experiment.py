import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gplearn.gplearn.genetic import SymbolicRegressor, SymbolicClassifier
from gplearn.gplearn.model import ShapeNN
from utils import create_df_from_cached_results, load_share_from_checkpoint, evaluate_shape, extract_slope_intercept
import torch
from sklearn.metrics import r2_score
from benchmarks import run_experiment, categorical_variables_per_dataset, create_categorical_variable_dict
from load_data import load_data

if __name__ == "__main__":

    device = 'cuda'
    n_jobs = 1
    population_size = 100
    generations = 10
    dataset_name = "breast_w"

    task = 'classification'
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
            'batch_size':2000,
            'shape_class':ShapeNN,
            'constructor_dict': constructor_dict_ShapeNN,
            'num_workers_dataloader': 0,
            'seed':42
            }
        }

    esr_parameter_dict = None
    if task == 'regression':
        esr = SymbolicRegressor(**gp_config, categorical_variables=create_categorical_variable_dict(dataset_name,task))
    elif task == 'classification':
        esr = SymbolicClassifier(**gp_config, categorical_variables=create_categorical_variable_dict(dataset_name,task))

    esr_score_mean, esr_score_std, model = run_experiment(dataset_name, esr, esr_parameter_dict, task, random_state=global_seed, return_model=True)
    print(esr_score_mean)
    print(esr_score_std)

    df = create_df_from_cached_results(model.cached_results)

    df.to_csv(f"{model.timestamp}_cached_results.csv")

