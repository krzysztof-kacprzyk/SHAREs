import sys
sys.path.append('../')
from experiments.risk_scores_data import generate_data
from sklearn.metrics import r2_score
from gplearn.gplearn.genetic import SymbolicRegressor
from gplearn.gplearn.model import ShapeNN
import matplotlib.pyplot as plt
import time
import torch
from experiments.utils import create_df_from_cached_results, load_share_from_checkpoint, get_n_shapes, get_n_variables
import pandas as pd
import pmlb

if __name__ == '__main__':

    df = pmlb.fetch_data('feynman_I_18_12',return_X_y=False)
    df = df.sample(200,random_state=0)
    X = df[['r','F','theta']].values
    y = df['target'].values
    X_train = X[:100,:]
    X_test = X[100:,:]
    y_train = y[:100]
    y_test = y[100:]

    def test_share(device, n_jobs, population_size, generations, batch_size):
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
                'shape_class':ShapeNN,
                'constructor_dict': constructor_dict_ShapeNN,
                'num_workers_dataloader': 0,
                'seed':42,
                'checkpoint_folder':'results/checkpoints',
                'keep_models':True
                }
            }

        esr = SymbolicRegressor(**gp_config, categorical_variables={})
        esr.fit(X_train,y_train)
        return esr
    
    t1 = time.time()
    population_size = 500
    generations = 10
    share = test_share('cpu',1,population_size,generations,1000)

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
        loss_train.append(r2_score(y_train,esr.predict(X_train)))
        loss_val.append(r2_score(y_test,esr.predict(X_test)))
    res_df['r2_val'] = loss_val
    res_df['r2_train'] = loss_train
    res_df['n_shapes'] = n_shapes
    res_df['n_variables'] = n_variables

    res_df.to_csv('results/Figure_5_results.csv')

    t2 = time.time()

    # Save the time it took to run the experiment
    with open('results/Figure_5_time.txt','w') as f:
        f.write(f'{t2-t1}')

    # Save the timestamp
    with open('results/Figure_5_timestamp.txt','w') as f:
        f.write(timestamp)
    




