import pandas as pd
import os
from gplearn.gplearn.genetic import SymbolicRegressor
from gplearn.gplearn._program import _Program
from gplearn.gplearn.model import ShapeNN, LitModel

from gplearn.gplearn.functions import _function_map, _Function
from gplearn.gplearn.fitness import _fitness_map
from gplearn.gplearn.utils import check_random_state
from gplearn.gplearn._program import _Program

import torch

import numpy as np

def evaluate_shape(shape,t):
    shape.to(torch.device('cpu'))
    t = torch.from_numpy(t).float()
    with torch.no_grad():
        shape.eval()
        return shape(t).numpy().flatten()
    
def extract_slope_intercept(shape,arg1,arg2):
    args = np.array([arg1,arg2])
    ys = evaluate_shape(shape,args)
    slope = (ys[1]-ys[0])/(args[1]-args[0])
    intercept = ys[0] - slope * args[0]
    return slope, intercept


def create_df_from_cached_results(results_dict):

    def get_n_shapes(program_list):
        res = 0
        for node in program_list:
            if node == 'shape':
                res += 1
        return res

    def get_n_variables(program_list):
        res = 0
        for node in program_list:
            if isinstance(node,int):
                res += 1
        return res

    programs = list(results_dict.keys())
    losses = [results_dict[k] for k in programs]
    num_of_shapes = [get_n_shapes(k) for k in programs]
    num_of_variables = [get_n_variables(k) for k in programs]
    return pd.DataFrame({'program':programs,
                            'loss':losses,
                            'num_of_shapes':num_of_shapes,
                            'num_of_variables':num_of_variables})



def load_share_from_checkpoint(timestamp, program, checkpoint_dir='checkpoints', task='regression',n_features=3,equation_id=None, categorical_variables_dict={}):
    # TODO: make it work in general, pickle configs and then load them
    if isinstance(program,str):
        # remove brackets
        program = program.replace('(',',')
        program = program.replace(')','')
        program = program.replace("'","")
        program = program.replace(" ","")
        program = program.replace("X","")
        program_list = program.split(',')
        new_program_list = []
        for node in program_list:
            if node == '':
                continue
            if node.isnumeric():
                new_program_list.append(int(node))
            else:
                new_program_list.append(_function_map[node])
        program_list = new_program_list

    elif isinstance(program,tuple):
        program_list = list(program)
        new_program_list = []
        for node in program_list:
            if isinstance(node,str):
                new_program_list.append(_function_map[node])
            else:
                new_program_list.append(node)
        program_list = new_program_list
    else:
        raise ValueError('program is neither a list nor a string')

    constructor_dict_ShapeNN = {
        'n_hidden_layers':5,
        'width':10,
        'activation_name':'ELU'
        }

    program_config = {
        'function_set' : [_function_map['add'],_function_map['sub'],_function_map['mul'],_function_map['div'],_function_map['shape']],
        'arities' : {1: [_function_map['shape']], 2: [_function_map['add'],_function_map['sub'],_function_map['mul'],_function_map['div']]},
        'init_depth' : (2,5),
        'init_method' : 'half and half',
        'n_features' : n_features,
        'const_range' : None,
        'metric' : _fitness_map['mse'],
        'p_point_replace' : 0.2,
        'parsimony_coefficient' : 0.0,
        'random_state': check_random_state(0),
        'optim_dict': {
            'alg':'adam',
            'lr': 1e-2,
            'max_n_epochs':1000,
            'tol':1e-3,
            'n_iter_no_change':10,
            'task':task,
            'device':'cpu',
            'batch_size':1000,
            'shape_class':ShapeNN,
            'constructor_dict': constructor_dict_ShapeNN,
            'num_workers_dataloader': 0,
            'seed':2,
            },
        'timestamp':timestamp
    }

    program = _Program(**program_config, program=program_list)
    program.categorical_variables_dict = categorical_variables_dict
    program.keys = sorted(categorical_variables_dict.keys())

    population_size = 500
    generations = 10
    global_seed = 42
    device = 'cpu'
    batch_size = 1000
    n_jobs = 1

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
            'seed':42
            }
        }

    program_str = str(program)
    
        
        
    if equation_id is None:
        full_path = os.path.join(checkpoint_dir,timestamp,f"{program_str}-best_val_loss.ckpt")
    else:
        full_path = os.path.join(checkpoint_dir,timestamp,f"{equation_id}-best_val_loss.ckpt")

    if program.is_fitting_necessary([]):
        # model = LitModel(program)
        # print(model)
        model = LitModel.load_from_checkpoint(full_path, program=program,strict=True)
        program.model = model

    esr = SymbolicRegressor(**gp_config, categorical_variables=categorical_variables_dict)
    esr._program = program
    esr.n_features_in_ = n_features

    return esr



    
    

