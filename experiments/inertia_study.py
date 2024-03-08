import sys
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
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
from experiments.load_data import load_data, generate_inertia_data
from datetime import datetime
import json
from copy import deepcopy
import argparse
from experiments.utils import create_df_from_cached_results, load_share_from_checkpoint, get_n_shapes, get_n_variables
import time



if __name__ == "__main__":

    t1 = time.time()

    simple = True

    num_of_variables = 2 if simple else 3

    # Generate data
    df = generate_inertia_data(200, seed=0, simple=simple)

    if simple:
        feature_columns = ['angle','object']
    else:
        feature_columns = ['angle','object','length']
    target_column = ['target']

    X = df[feature_columns]
    y = df[target_column]

    df_test = generate_inertia_data(200, seed=1, simple=simple)
    X_test = df_test[feature_columns]
    y_test = df_test[target_column]
    categorical_variables = [1]
    is_cat = np.array([i in categorical_variables for i in range(len(X.columns))])

    cat_cols = X.columns.values[is_cat]
    num_cols = X.columns.values[~is_cat]

    categories = []
    for i in categorical_variables:
        uniques = list(X.iloc[:,i].unique())
        categories.append(sorted(uniques))
    
    cat_pipe = Pipeline([('ordinal', OrdinalEncoder(categories=categories))])
    num_pipe = Pipeline([('std',StandardScaler())])
    # num_pipe = Pipeline([('identity', FunctionTransformer(lambda x: x.values))])
    transformers = [
        ('cat', cat_pipe, cat_cols),
        ('num', num_pipe, num_cols)
    ]
    ct = ColumnTransformer(transformers=transformers)
    
    processing = Pipeline([
        ('ct',ct),
    ])

    y_scaler = StandardScaler()

     # Fit and transform training data
    X_train = processing.fit_transform(X)
    y_train = y_scaler.fit_transform(y)[:,0]

    # Transform test data
    X_test = processing.transform(X_test)
    y_test = y_scaler.transform(y_test)[:,0]

    dataset_name = 'inertia_simple' if simple else 'inertia'
    task = 'regression'
    device = 'cuda'
    n_jobs = 1
    population_size = 100
    generations = 10
    global_seed = 42
    n_trials = 1
    start_from_trial = 0

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
            'max_n_epochs':400,
            'tol':1e-3,
            'task':task,
            'device':device,
            'batch_size':64,
            'shape_class':ShapeNN,
            'constructor_dict': constructor_dict_ShapeNN,
            'num_workers_dataloader': 0,
            'seed':42,
            'checkpoint_folder': 'results/checkpoints',
            'keep_models':True,
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

    timestamp = model.timestamp

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
        n_shapes.append(get_n_shapes(eq, categorical_variables=[1]))
        n_variables.append(get_n_variables(eq))
        esr = load_share_from_checkpoint(timestamp, eq, checkpoint_dir='results/checkpoints', task='regression',n_features=num_of_variables, equation_id=id, categorical_variables_dict=create_categorical_variable_dict(dataset_name,task))
        loss_train.append(r2_score(y_train,esr.predict(X_train)))
        loss_val.append(r2_score(y_test,esr.predict(X_test)))
    res_df['r2_val'] = loss_val
    res_df['r2_train'] = loss_train
    res_df['n_shapes'] = n_shapes
    res_df['n_variables'] = n_variables

    res_df.to_csv(f'results/Inertia_results.csv')

    t2 = time.time()

    # Save the time it took to run the experiment
    with open(f'results/Inertia_time.txt','w') as f:
        f.write(f'{t2-t1}')

    # Save the timestamp
    with open(f'results/Inertia_timestamp.txt','w') as f:
        f.write(timestamp)
    

