import sys
sys.path.append('../')

import numpy as np
import matplotlib.pyplot as plt
from experiments.benchmarks import run_experiment
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from collections import defaultdict
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
import time

def get_model(model,task):
    if model == 'xgb':
        if task == 'regression':
            return XGBRegressor()
        elif task == 'classification':
            return XGBClassifier()
    elif model == 'ebm':
        if task == 'regression':
            return ExplainableBoostingRegressor()
        elif task == 'classification':
            return ExplainableBoostingClassifier()
    elif model == 'ebm_no_interactions':
        if task == 'regression':
            return ExplainableBoostingRegressor(interactions=0)
        elif task == 'classification':
            return ExplainableBoostingClassifier(interactions=0)
        

task = 'regression'
dataset_names = [
    'feynman_I_6_2b',
    'feynman_I_8_14',
    'feynman_I_12_2',
    'feynman_I_12_11',
    'feynman_I_18_12',
    'feynman_I_29_16',
    'feynman_I_32_5',
    'feynman_I_40_1',
    'feynman_II_2_42'
]

global_seed = 0
model_names = ['xgb','ebm_no_interactions','ebm']

# start timer
start = time.time()

results = defaultdict(list)
for dataset_name in dataset_names:
    results['dataset_name'].append(dataset_name)
    for model_name in model_names:
        model = get_model(model_name,task)
        t1 = time.time()
        score_mean, score_std = run_experiment(dataset_name, model, None, task, random_state=global_seed)
        t2 = time.time()
        results[f'{model_name}_mean'].append(score_mean)
        results[f'{model_name}_std'].append(score_std)
        results[f'{model_name}_time'].append(t2-t1)
        
df = pd.DataFrame(results)

# end timer
end = time.time()
    
df.to_csv('results/feynman_benchmark_results.csv')

# Save the time to a file
with open('results/Table_2_time.txt', 'w') as f:
    f.write(str(end - start))