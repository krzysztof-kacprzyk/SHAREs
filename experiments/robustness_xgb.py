import sys
sys.path.append('../')
import pickle
import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import clone
from sklearn.model_selection import train_test_split
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
import os


from temperature import generate_data
from xgboost import XGBRegressor

if __name__ == '__main__':

    # Argparse get a list of noise levels
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise_levels', nargs='+', type=float)
    parser.add_argument('--tune', action='store_true')
    args = parser.parse_args()
    noise_levels = args.noise_levels

    global_seed = 42

    # Generate seeds for 3 runs using a generator
    gen = np.random.default_rng(global_seed)
    all_random_seeds = gen.choice(10000, 1000, replace=False)
    random_seeds = all_random_seeds[:3]

    if noise_levels is None:
        noise_levels = [0.0]

    df_results = pd.DataFrame(columns=['run','seed','noise','r2'])

    global_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for noise in noise_levels:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        # Generate data
        df = generate_data(1000, (1,800), (1.0,4.0), (-100,0), seed=0, noise=noise)

        feature_columns = ['energy','mass','initial_temp']
        target_column = 'temperature'

        X = df[feature_columns].values
        y = df[target_column].values

        df_test = generate_data(1000, (1,800), (1.0,4.0), (-100,0), seed=1, noise=noise)
        X_test = df_test[feature_columns].values
        y_test = df_test[target_column].values

        org_model = XGBRegressor()

        if args.tune:
            # Hyperparameter tuning using optuna

            print("Hyperparameter tuning")

            hyperparam_save_path = os.path.join('results','tuning',f'temperature_xgb_{timestamp}.json')
            study_save_path = os.path.join('results','tuning',f'study_{timestamp}.pkl')
            study_df_save_path = os.path.join('results','tuning',f'study_df_{timestamp}.csv')


            # Divide training data into training and validation sets

            X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(X, y, test_size=0.2, random_state=global_seed)
            
            parameter_dict = lambda trial: {
                'n_estimators': trial.suggest_int('n_estimators', 10, 1000, log=True),
                'eta': trial.suggest_float('eta', 1e-2, 10.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.2, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
                'lambda': trial.suggest_float('lambda', 1e-1, 10.0, log=True), 
            }

            def objective(trial):
                # Create a hyperparameter space
                params = parameter_dict(trial)
                # Instantiate the model with the hyperparameters
                model = clone(org_model)
                model.set_params(random_state=global_seed)
                model.set_params(**params)

                model.fit(X_train_tune, y_train_tune)
                # Evaluate the model
                y_pred = model.predict(X_val_tune)
                # if any nan in y_pred, return -inf
                if np.isnan(y_pred).any():
                    return -1e9
                score = r2_score(y_val_tune,y_pred)
            
                return score
        
            # Create a study object and optimize the objective function.
            sampler = optuna.samplers.TPESampler(seed=global_seed, multivariate=True)
            study = optuna.create_study(sampler=sampler,direction='maximize')
            study.optimize(objective, n_trials=100)
            best_trial = study.best_trial
            best_hyperparameters = best_trial.params

            print('[Best hyperparameter configuration]:')
            print(best_hyperparameters)

            # Save best hyperparameters
            with open(hyperparam_save_path, 'w') as f:
                json.dump(best_hyperparameters, f)
            
            # Save optuna study
            with open(study_save_path, 'wb') as f:
                pickle.dump(study, f)

            # Save trials dataframe
            df_study = study.trials_dataframe()
            df_study.set_index('number', inplace=True)
            df_study.to_csv(study_df_save_path)

            print("Hyperparameter tuning done")

        for run, seed in enumerate(random_seeds):
            
            model = clone(org_model)
            model.set_params(random_state=seed)

            if args.tune:
                model.set_params(**best_hyperparameters)

            model.fit(X, y)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            # Add result to df using concat
            df_results = pd.concat([df_results, pd.DataFrame({'run':[run],'seed':[seed],'noise':[noise], 'r2':[r2]})], ignore_index=True)

            # Save df
            filepath = f'results/robustness/xgb_{global_timestamp}.csv'
            
            # Check if folder exists and create if not
            import os
            if not os.path.exists('results/robustness'):
                os.makedirs('results/robustness')
            
            df_results.to_csv(filepath)