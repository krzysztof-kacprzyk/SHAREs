import sys
sys.path.append('../')
import json
import os
from pmlb import fetch_data
from pygam import LinearGAM, LogisticGAM
from sklearn.model_selection import KFold, ParameterGrid, ParameterSampler, StratifiedKFold, train_test_split, StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score, r2_score
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC, SVR
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
import numpy as np
from gplearn.gplearn.genetic import SymbolicRegressor, SymbolicClassifier
from gplearn.gplearn.model import ShapeNN
import time
from datetime import datetime
import pickle
import argparse
import operator
import functools
from tqdm import tqdm
import pandas as pd
import optuna

import scipy

from experiments.load_data import load_data

def generate_indices(n, train_size, val_size, seed=0):
    gen = np.random.default_rng(seed)
    train_indices = gen.choice(n, int(n*train_size), replace=False)
    train_indices = [i.item() for i in train_indices]
    val_indices = gen.choice(list(set(range(n)) - set(train_indices)), int(n*val_size), replace=False)
    val_indices = [i.item() for i in val_indices]
    test_indices = list(set(range(n)) - set(train_indices) - set(val_indices))
    return train_indices, val_indices, test_indices


regression_dataset_names = [
    '192_vineyard',
    '228_elusage',
    '712_chscase_geyser1',
    '210_cloud',
    'nikuradse_1',
    '678_visualizing_environmental',
    '519_vinnie',
    '690_visualizing_galaxy',
    '529_pollen',
    '1201_BNG_breastTumor',
    '218_house_8L',
    '537_houses',
    '1193_BNG_lowbwt',
    '1199_BNG_echoMonths',
    '1203_BNG_pwLinear',
    '1595_poker',
    '225_puma8NH',
    '344_mv',
    '564_fried',
    'biomed',
    '522_pm10',
    '547_no2',
    '561_cpu',
    'yeast',
    'ecoli',
    'wine_quality_white',
    'wine_quality_red',
    'feynman_I_6_2b',
    'feynman_I_8_14',
    'feynman_I_9_18',
    'feynman_I_10_7',
    'feynman_I_12_2',
    'feynman_I_12_4',
    'feynman_I_12_11',
    'feynman_I_13_12',
    'feynman_I_14_3',
    'feynman_I_18_12',
    'feynman_I_24_6',
    'feynman_I_29_16',
    'feynman_I_30_5',
    'feynman_I_32_5',
    'feynman_I_34_8',
    'feynman_I_34_1',
    'feynman_I_37_4',
    'feynman_I_38_12',
    'feynman_I_39_11',
    'feynman_I_39_22',
    'feynman_I_40_1',
    'feynman_I_43_16',
    'feynman_I_43_31',
    'feynman_I_44_4',
    'feynman_I_47_23',
    'feynman_II_2_42',
    'feynman_II_3_24',
    'feynman_II_4_23',
    'feynman_II_6_11',
    'feynman_II_6_15a',
    'feynman_II_6_15b',
    'feynman_II_8_7',
    'feynman_II_10_9',
    'feynman_II_11_3',
    'feynman_II_11_20',
    'feynman_II_13_17',
    'feynman_II_13_23',
    'feynman_II_15_4',
    'feynman_II_15_5',
    'feynman_II_21_32',
    'feynman_II_24_17',
    'feynman_II_27_16',
    'feynman_II_27_18',
    'feynman_II_34_2a',
    'feynman_II_34_2',
    'feynman_II_34_11',
    'feynman_II_34_29a',
    'feynman_II_34_29b',
    'feynman_II_36_38',
    'feynman_II_37_1',
    'feynman_II_38_3',
    'feynman_III_4_32',
    'feynman_III_7_38',
    'feynman_III_8_54',
    'feynman_III_10_19',
    'feynman_III_13_18',
    'feynman_III_14_14',
    'feynman_III_15_12',
    'feynman_III_15_14',
    'feynman_III_15_27',
    'feynman_III_17_37',
    'feynman_III_19_51',
    'feynman_III_21_20',
]
classification_dataset_names = [
    'magic',
    'banana',
    'adult',
    'australian',
    'breast_w',
    'breast',
    'diabetes',
    'flare',
    'irish',
    'phoneme',
    'pima',
    'profb',
    'breast_cancer'
]
categorical_variables_per_dataset = {
    '192_vineyard':[],
    '228_elusage':[],
    '712_chscase_geyser1':[],
    '210_cloud':[0,1],
    'nikuradse_1':[0],
    '678_visualizing_environmental':[],
    '519_vinnie':[0],
    '690_visualizing_galaxy':[],
    '529_pollen':[],
    '1201_BNG_breastTumor':[1,3,4,5,7,8],
    'magic':[],
    'banana':[],
    '218_house_8L':[],
    '537_houses':[],
    '1193_BNG_lowbwt':[0,3,4,5,6,7],
    '1199_BNG_echoMonths':[0,2,8],
    '1203_BNG_pwLinear':[0,1,2,3,4,5,6,7,8,9],
    '1595_poker':[0,2,4,6,8],
    '225_puma8NH':[],
    '344_mv':[2,6,7],
    '564_fried':[],
    'adult':[1,3,5,6,7,8,9,13],
    'australian':[0,3,7,8,10,11],
    'breast_w':[],
    'breast':[],
    'diabetes':[],
    'flare':[2,3,4,5,6,7,8],
    'irish':[0,4],
    'phoneme':[],
    'pima':[],
    'profb':[0,6,8],
    'breast_cancer':[1,4,5,6,8],
    'biomed':[],
    '522_pm10':[],
    '547_no2':[],
    '561_cpu':[],
    'yeast':[4,5],
    'ecoli':[2],
    'concrete':[],
    'wine_quality_white':[],
    'wine_quality_red':[],
    'servo':[0,1],
    'yacht':[],
    'energy_efficiency_1':[5,7],
    'energy_efficiency_2':[5,7],
    'boston':[3,8],
    'california':[],
    'feynman_I_6_2b':[],
    'feynman_I_8_14':[],
    'feynman_I_9_18':[],
    'feynman_I_10_7':[],
    'feynman_I_12_2':[],
    'feynman_I_12_4':[],
    'feynman_I_12_11':[],
    'feynman_I_13_12':[],
    'feynman_I_14_3':[],
    'feynman_I_18_12':[],
    'feynman_I_24_6':[],
    'feynman_I_29_16':[],
    'feynman_I_30_5':[],
    'feynman_I_32_5':[],
    'feynman_I_34_8':[],
    'feynman_I_34_1':[],
    'feynman_I_37_4':[],
    'feynman_I_38_12':[],
    'feynman_I_39_11':[],
    'feynman_I_39_22':[],
    'feynman_I_40_1':[],
    'feynman_I_43_16':[],
    'feynman_I_43_31':[],
    'feynman_I_44_4':[],
    'feynman_I_47_23':[],
    'feynman_II_2_42':[],
    'feynman_II_3_24':[],
    'feynman_II_4_23':[],
    'feynman_II_6_11':[],
    'feynman_II_6_15a':[],
    'feynman_II_6_15b':[],
    'feynman_II_8_7':[],
    'feynman_II_10_9':[],
    'feynman_II_11_3':[],
    'feynman_II_11_20':[],
    'feynman_II_13_17':[],
    'feynman_II_13_23':[],
    'feynman_II_15_4':[],
    'feynman_II_15_5':[],
    'feynman_II_21_32':[],
    'feynman_II_24_17':[],
    'feynman_II_27_16':[],
    'feynman_II_27_18':[],
    'feynman_II_34_2a':[],
    'feynman_II_34_2':[],
    'feynman_II_34_11':[],
    'feynman_II_34_29a':[],
    'feynman_II_34_29b':[],
    'feynman_II_36_38':[],
    'feynman_II_37_1':[],
    'feynman_II_38_3':[],
    'feynman_III_4_32':[],
    'feynman_III_7_38':[],
    'feynman_III_8_54':[],
    'feynman_III_10_19':[],
    'feynman_III_13_18':[],
    'feynman_III_14_14':[],
    'feynman_III_15_12':[],
    'feynman_III_15_14':[],
    'feynman_III_15_27':[],
    'feynman_III_17_37':[],
    'feynman_III_19_51':[],
    'feynman_III_21_20':[],
    'stress_strain':[],
    'temperature':[],
    'inertia':[1],
    'inertia_simple':[1],
}
non_pmlb_datasets = {
    'regression':['concrete','servo','yacht','energy_efficiency_1','energy_efficiency_2','boston','california','stress_strain','temperature','inertia','inertia_simple'],
    'classification':[]
}


def score(model, X, y_true, task, raw=False):
    if task == 'regression':
        y_pred = model.predict(X)
        if raw:
            res = mean_squared_error(y_true,y_pred)
        else:
            res = r2_score(y_true,y_pred)
        # print(f"{model} | score: {score}")
    elif task == 'classification':
        if isinstance(model, LogisticGAM):
            y_pred_proba = model.predict_proba(X)
        else:
            y_pred_proba = model.predict_proba(X)[:, 1]
        if raw:
            res = log_loss(y_true,y_pred)
        else:
            res = roc_auc_score(y_true,y_pred_proba)
        # print(f"{model} | score: {score}")
    return res

def find_extremum(operation, itera, min_or_max):
    ex_value = None
    comp = operator.gt if min_or_max == 'max' else operator.lt
    for i in itera:
        obj, val = operation(i)
        if ex_value is None:
            ex_value = val
            ex_object = obj
        else:
            if comp(val,ex_value):
                ex_value = val
                ex_object = obj
    return (ex_object, ex_value)

def load_df(dataset_name,task):
    if dataset_name in non_pmlb_datasets[task]:
        dataset = load_data(dataset_name)
    else:
        dataset = fetch_data(dataset_name, return_X_y=False)
        if dataset_name == 'diabetes':
            dataset['target'] = (dataset['target'] == 2)
    return dataset

def create_categorical_variable_dict(dataset_name,task):
    dataset = load_df(dataset_name,task)
    X = dataset.drop(columns=['target'])
    categorical_variables = categorical_variables_per_dataset[dataset_name]
    categories = {}
    for index, i in enumerate(categorical_variables):
        n_uniques = len(X.iloc[:,i].unique())
        categories[index] = n_uniques
    return categories



def run_experiment(dataset_name, org_model, parameter_dict, task, random_state, return_model=False, disable_ohe=False, timestamp=None, n_trials=10, start_from_trial=0):

    # Create random seeds using numpy random generator
    gen = np.random.default_rng(random_state)
    all_random_seeds = gen.choice(10000, 1000, replace=False)
    random_seeds = all_random_seeds[:n_trials]

    print(random_seeds)

    df = pd.DataFrame(columns=['dataset','model','fold','score','time','timestamp'])

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    

    model_name = org_model.__class__.__name__
    if model_name == 'ExplainableBoostingClassifier' or model_name == 'ExplainableBoostingRegressor':
        if org_model.interactions == 0:
            model_name += 'No'

    file_name = f"{dataset_name}_{model_name}_{timestamp}.csv"


    file_path = os.path.join('results',file_name)

    hyperparam_save_path = os.path.join('results','tuning',f'{dataset_name}_{model_name}_{timestamp}.json')
    study_save_path = os.path.join('results','tuning',f'study_{timestamp}.pkl')
    study_df_save_path = os.path.join('results','tuning',f'study_df_{timestamp}.csv')


    if not isinstance(org_model, LinearRegression): 
        org_model.set_params(random_state=random_state)
    # random_state = check_random_state(random_state)
    dataset = load_df(dataset_name,task)
    X = dataset.drop(columns=['target'])
    y = dataset[['target']]

    # y_max = y.max()
    # y_min = y.min()

    # y = (y-y_min)/(y_max-y_min)

    # First, we choose a validation set for hyper-parameter tuning
    # if task == 'classification':
    #     sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
    #     train_index, val_index = next(sss.split(X,y))
    #     X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    #     y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    # elif task == 'regression':
    #     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=random_state)

    categorical_variables = categorical_variables_per_dataset[dataset_name]
    is_cat = np.array([i in categorical_variables for i in range(len(X.columns))])

    cat_cols = X.columns.values[is_cat]
    num_cols = X.columns.values[~is_cat]

    if isinstance(org_model, SymbolicRegressor) or isinstance(org_model, SymbolicClassifier) or disable_ohe:
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
    else:
        cat_ohe_step = ('ohe', OneHotEncoder(sparse=False,
                                            handle_unknown='ignore'))

        cat_pipe = Pipeline([cat_ohe_step])
        num_pipe = Pipeline([('identity', FunctionTransformer())])
        transformers = [
            ('cat', cat_pipe, cat_cols),
            ('num', num_pipe, num_cols)
        ]
        ct = ColumnTransformer(transformers=transformers)
        
        processing = Pipeline([
            ('ct',ct),
            ('std',StandardScaler())
        ])

    if task == 'classification':
        # Identity transformer for classification
        y_scaler = FunctionTransformer(lambda x: x.values)
    else:
        y_scaler = StandardScaler()

    # Split dataset into train and test sets
    if task == 'classification':
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_index, test_index = next(sss.split(X,y))
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    elif task == 'regression':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Fit and transform training data
    X_train = processing.fit_transform(X_train)
    y_train = y_scaler.fit_transform(y_train)[:,0]

    # Transform test data
    X_test = processing.transform(X_test)
    y_test = y_scaler.transform(y_test)[:,0]


    if parameter_dict is not None:
        # Hyperparameter tuning using optuna

        print("Hyperparameter tuning")

        # Divide training data into training and validation sets

        if task == 'classification':
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
            train_index, val_index = next(sss.split(X_train,y_train))
            X_train_tune, X_val_tune = X_train[train_index], X_train[val_index]
            y_train_tune, y_val_tune = y_train[train_index], y_train[val_index]
        elif task == 'regression':
            X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(X_train, y_train, test_size=0.2, random_state=random_state)
        
        def objective(trial):
            # Create a hyperparameter space
            params = parameter_dict(trial)
            # Instantiate the model with the hyperparameters
            if isinstance(org_model, LinearGAM):
                model = LinearGAM()
            else:
                model = clone(org_model)

            # Do not set random_state for PyGAM
            if not isinstance(org_model, LinearGAM) and not isinstance(org_model, LogisticGAM):
                model.set_params(random_state=random_state)
            model.set_params(**params)

            model.fit(X_train_tune, y_train_tune)
            # Evaluate the model
            if task == 'regression':
                y_pred = model.predict(X_val_tune)
                score = r2_score(y_val_tune,y_pred)
            elif task == 'classification':
                # If pygam then the result is 1d
                if isinstance(org_model, LogisticGAM):
                    y_pred_proba = model.predict_proba(X_val_tune)
                else:
                    y_pred_proba = model.predict_proba(X_val_tune)[:, 1]
                score = roc_auc_score(y_val_tune,y_pred_proba)
            return score
        
        # Create a study object and optimize the objective function.
        sampler = optuna.samplers.TPESampler(seed=random_state)
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
    
    scores = []

    for i, random_seed in enumerate(random_seeds):
        print(f"Trial {i} | Random seed: {random_seed}")
        if i < start_from_trial:
            print("Skipping trial")
            continue
        time_start = time.time()

        # if linear gam then instantiate a new model
        if isinstance(org_model, LinearGAM):
            model = LinearGAM()
        else:
            model = clone(org_model)

        if parameter_dict is not None:
            model.set_params(**best_hyperparameters)

        # If not pygam then set random state
        if not isinstance(org_model, LinearGAM) and not isinstance(org_model, LogisticGAM):
            model.set_params(random_state=random_seed)
        # if isinstance(org_model, XGBRegressor):
        #     model = XGBRegressor(random_state=random_seed)
        # elif isinstance(org_model, XGBClassifier):
        #     model = XGBClassifier(random_state=random_seed)

    
        if isinstance(org_model, SymbolicRegressor) or isinstance(org_model, SymbolicClassifier):
            if i == len(random_seeds)-1:
                model.optim_dict['keep_models'] = True
            else:
                model.optim_dict['keep_models'] = False

        model.fit(X_train, y_train)
        new_score = score(model,X_test,y_test,task)
        scores.append(new_score)
        print(scores)

        time_end = time.time()

        # Check if model has timestamp parameter
        if hasattr(model, 'timestamp'):
            fold_timestamp = model.timestamp
        else:
            fold_timestamp = "Not available"

        # Create a new dataframe row
        new_df_row = pd.DataFrame({
            'global_seed': [random_state],
            'trial_seed': [random_seed],
            'dataset': [dataset_name],
            'model': [model_name],
            'fold': [i],
            'score': [new_score],
            'time': [time_end-time_start],
            'timestamp':[fold_timestamp]
        })
        # Append the new row to the dataframe
        df = pd.concat([df, new_df_row], ignore_index=True)
        df.to_csv(file_path)
        
    if return_model:
        return np.mean(scores), np.std(scores), model
    return np.mean(scores), np.std(scores)

      

   

# def run_experiment_sr(dataset_name, gp_dict, opt_dict, task, parsimony_coefficients=None, seed=0, return_model=False, verbose=False):
    # print("SR")
    # if dataset_name in non_pmlb_datasets[task]:
    #     dataset = load_data(dataset_name)
    # else:
    #     dataset = fetch_data(dataset_name, return_X_y=False)
    # X = dataset.drop(columns=['target'])
    # y = dataset['target']

    # y_max = y.max()
    # y_min = y.min()

    # y = (y-y_min)/(y_max-y_min)
    
    # if task == 'classification':
    #     sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    #     train_index, test_index = next(sss.split(X,y))
    #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # elif task == 'regression':
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # categorical_variables = categorical_variables_per_dataset[dataset_name]
    # is_cat = np.array([i in categorical_variables for i in range(len(X.columns))])
   
    # # One Hot Encoding of the categorical data
    # cat_cols = X.columns.values[is_cat]
    # num_cols = X.columns.values[~is_cat]

    # cat_pipe = Pipeline([('ordinal', OrdinalEncoder())])
    # num_pipe = Pipeline([('std',StandardScaler())])
    # transformers = [
    #     ('cat', cat_pipe, cat_cols),
    #     ('num', num_pipe, num_cols)
    # ]
    # ct = ColumnTransformer(transformers=transformers)
    
    # processing = Pipeline([
    #     ('ct',ct),
    # ])
    
    # # X_train = processing.fit_transform(X_train)
    # # X_test = processing.transform(X_test)
    
    # # scores = {}

    # if parsimony_coefficients is None:

    #     if task == 'regression':
    #         sr = SymbolicRegressor(**gp_dict,opt_dict=opt_dict,random_state=seed,verbose=verbose,categorical_variables=categorical_variables)
    #     elif task == 'classification':
    #         sr = SymbolicClassifier(**gp_dict,opt_dict=opt_dict,random_state=seed,verbose=verbose,categorical_variables=categorical_variables)

        
    #     X_train = processing.fit_transform(X_train)

    #     sr.fit(X_train, y_train)

    #     best_estimator = sr
    
    # else:
    #     scores_tuning = []

    #     if task == 'classification':
    #         sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    #         val_train_index, val_test_index = next(sss.split(X_train,y_train))
    #         X_val_train, X_val_test = X_train.iloc[val_train_index], X_train.iloc[val_test_index]
    #         y_val_train, y_val_test = y.iloc[val_train_index], y.iloc[val_test_index]
    #     elif task == 'regression':
    #         X_val_train, X_val_test, y_val_train, y_val_test = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

    #     X_val_train = processing.fit_transform(X_val_train)
    #     X_val_test = processing.transform(X_val_test)

    #     for parsimony_coefficient in parsimony_coefficients:
    #         print(f"Parsimony coefficient: {parsimony_coefficient}")
    #         if task == 'regression':
    #             sr = SymbolicRegressor(parsimony_coefficient=parsimony_coefficient,**gp_dict,opt_dict=opt_dict,random_state=seed,categorical_variables=categorical_variables)
    #             sr.fit(X_val_train, y_val_train)
    #             y_val_pred = sr.predict(X_val_test)
    #             score = r2_score(y_val_test,y_val_pred)
    #         elif task == 'classification':
    #             sr = SymbolicClassifier(parsimony_coefficient=parsimony_coefficient,**gp_dict,opt_dict=opt_dict,random_state=seed,categorical_variables=categorical_variables)
    #             sr.fit(X_val_train, y_val_train)
    #             y_val_pred_proba = sr.predict_proba(X_val_test)[:, 1]
    #             score = roc_auc_score(y_val_test,y_val_pred_proba)
    #         scores_tuning.append(score)
        
    #     max_ind = np.argmax(scores_tuning)
    #     best_parsimony_coefficient = parsimony_coefficients[max_ind]
    #     print(f"Best parsimony coefficient: {best_parsimony_coefficient}")
    #     if task == 'regression':
    #         best_estimator = SymbolicRegressor(parsimony_coefficient=best_parsimony_coefficient,**gp_dict,opt_dict=opt_dict,random_state=seed,categorical_variables=categorical_variables)
    #     elif task == 'classification':
    #         best_estimator = SymbolicClassifier(parsimony_coefficient=best_parsimony_coefficient,**gp_dict,opt_dict=opt_dict,random_state=seed,categorical_variables=categorical_variables)

    #     X_train = processing.fit_transform(X_train)
    #     best_estimator.fit(X_train, y_train)

    
    
    # X_test = processing.transform(X_test)
        
    # if task == 'regression':
    #     y_pred = best_estimator.predict(X_test)
    #     # score = np.sqrt(mean_squared_error(y_test,y_pred))
    #     score = r2_score(y_test,y_pred)
    #     # print(f"{dataset_name} | {model} | RMSE: {score}")
        
        
    # elif task == 'classification':
    #     y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]
    #     score = roc_auc_score(y_test,y_pred_proba)
    #     # print(f"{dataset_name} | {model} | ROC-AUC: {score}")
    
    # if return_model:
    #     return best_estimator, score
    # else:
    #     return score


class DictionaryDistribution():

    def __init__(self,param_grid):
        self.param_grid = param_grid

    def rvs(self,random_state):
        # Always sort the keys of a dictionary, for reproducibility
        items = sorted(self.param_grid.items())
        params = dict()
        for k, v in items:
            if hasattr(v, "rvs"):
                params[k] = v.rvs(random_state=random_state)
            else:
                params[k] = v[random_state.randint(len(v))]
        return params



def save(filename, log):
    with open(filename,"wb") as file:
        pickle.dump(log,file)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--algs", nargs='+', default=[])
    # parser.add_argument("--non_pmlb", action='store_true', default=False)
    parser.add_argument("--datasets", nargs='+', default=None)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--tune", action='store_true', default=False)
    args = parser.parse_args()



    global_seed = 42
    np.random.seed(global_seed)

    dt = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
    filename = f"results/benchmarks_{dt}_results.p"

    print(f"Results will be saved to {filename}")

    # if args.non_pmlb:
    #     dataset_names = non_pmlb_datasets
    # else:
    #     dataset_names = {'regression':regression_dataset_names,
    #                     'classification':classification_dataset_names}
    
    dataset_names = {}
    dataset_names['regression'] = regression_dataset_names + non_pmlb_datasets['regression']
    dataset_names['classification'] = classification_dataset_names + non_pmlb_datasets['classification']

    if args.datasets is not None:
        new_dataset_names = {'regression':[],'classification':[]}
        for dataset in args.datasets:
            if dataset in dataset_names['regression']:
                new_dataset_names['regression'].append(dataset)
            elif dataset in dataset_names['classification']:
                new_dataset_names['classification'].append(dataset)
        
        dataset_names = new_dataset_names

    log = {}

    for task in ['regression', 'classification']:
        print(f"Task: {task}")
        for dataset_name in dataset_names[task]:
            print(f"Dataset: {dataset_name}")

            if dataset_name not in log:
                log[dataset_name] = {}

            if 'pygam' in args.algs:
                pygam_start = time.time()
                if task == 'regression':
                    pygam = LinearGAM()
                elif task == 'classification':
                    pygam = LogisticGAM()
                
                pygam_parameter_dict = lambda trial: {
                    'lam': trial.suggest_float('lam', 1e-3, 10.0, log=True),
                    'n_splines': trial.suggest_int('n_splines', 1, 30),
                    'max_iter': trial.suggest_int('max_iter', 10, 1000, log=True)
                }

                pygam_score = run_experiment(dataset_name, pygam, pygam_parameter_dict, task, global_seed, n_trials=args.n_trials)
                pygam_end = time.time()
                log[dataset_name]['pygam'] = {'score':pygam_score, 'time':pygam_end-pygam_start}
                print(f"pygam | score: {pygam_score} | time: {pygam_end-pygam_start}")
                save(filename,log)

            if 'elastic' in args.algs:
                elastic_start = time.time()
                if task == 'regression':
                    elastic = ElasticNet()
                elif task == 'classification':
                    elastic = LogisticRegression(penalty='elasticnet',solver='saga')
                
                if task == 'regression':
                    elastic_parameter_dict = lambda trial: { 
                        'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True), 
                        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0)
                    }
                elif task == 'classification':
                    elastic_parameter_dict = lambda trial: { 
                        'C': trial.suggest_float('C', 1e-3, 10.0, log=True), 
                        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0)
                    }
                
                elastic_score = run_experiment(dataset_name, elastic, elastic_parameter_dict, task, global_seed, n_trials=args.n_trials)
                elastic_end = time.time()
                log[dataset_name]['elastic'] = {'score':elastic_score, 'time':elastic_end-elastic_start}
                print(f"elastic | score: {elastic_score} | time: {elastic_end-elastic_start}")
                save(filename,log)

            if 'lr' in args.algs:
                lr_start = time.time()
                if task == 'regression':
                    lr = LinearRegression()
                elif task == 'classification':
                    lr = LogisticRegression()
                # lr_parameter_dict = {}
                lr_parameter_dict = None
                lr_score = run_experiment(dataset_name, lr, lr_parameter_dict, task, global_seed, n_trials=args.n_trials)
                lr_end = time.time()
                log[dataset_name]['lr'] = {'score':lr_score, 'time':lr_end-lr_start}
                print(f"lr | score: {lr_score} | time: {lr_end-lr_start}")
                save(filename,log)

            if 'xgb' in args.algs:
                xgb_start = time.time()
                if task == 'regression':
                    xgb = XGBRegressor(colsample_bytree=0.5)
                elif task == 'classification':
                    xgb = XGBClassifier(colsample_bytree=0.5)

                # xgb_parameter_dict = {'max_depth': [3, 5, 6, 10, 15, 20],
                #             'subsample': np.arange(0.5, 1.0, 0.1),
                #             'n_estimators' : [100, 250, 500, 1000],
                #             'learning_rate' : [0.001,0.01, 0.05, 0.1, 0.2,0.3],
                #             'gamma' : [0,0.1,0.2,0.3,0.4],
                #             'colsample_bytree': np.arange(0.4, 1.0, 0.1),
                #             'colsample_bylevel': np.arange(0.4, 1.0, 0.1)}
                xgb_parameter_dict = lambda trial: {
                    'n_estimators': trial.suggest_int('n_estimators', 10, 1000, log=True),
                    'eta': trial.suggest_float('eta', 1e-3, 1.0, log=True),
                    'max_depth': trial.suggest_int('max_depth', 1, 10),
                    'gamma': trial.suggest_float('gamma', 1e-3, 10.0, log=True),
                    'subsample': trial.suggest_float('subsample', 0.2, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
                    'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
                    'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True)
                }
                if not args.tune:
                    xgb_parameter_dict = None
                xgb_score = run_experiment(dataset_name, xgb, xgb_parameter_dict, task, global_seed, n_trials=args.n_trials)
                xgb_end = time.time()
                log[dataset_name]['xgb'] = {'score':xgb_score, 'time':xgb_end-xgb_start}
                print(f"xgb | score: {xgb_score} | time: {xgb_end-xgb_start}")
                save(filename,log)

            if 'svm' in args.algs:
                svm_start = time.time()
                if task == 'regression': 
                    svm = SVR()
                elif task == 'classification':
                    svm = SVC()
                # svm_parameter_dict = {'C':[0.001,0.01,0.05,0.1,0.5,1.0,5.0,10.0,100.0]}
                svm_parameter_dict = None
                svm_score = run_experiment(dataset_name, svm, svm_parameter_dict, task, global_seed, n_trials=args.n_trials)
                svm_end = time.time()
                log[dataset_name]['svm'] = {'score':svm_score, 'time':svm_end-svm_start}
                print(f"svm | score: {svm_score} | time: {svm_end-svm_start}")
                save(filename,log)

            if 'ebm' in args.algs:
                ebm_start = time.time()
                if task == 'regression':
                    ebm = ExplainableBoostingRegressor()
                elif task == 'classification':
                    ebm = ExplainableBoostingClassifier()
                # ebm_parameter_dict = {'binning':['uniform','quantile','quantile_humanized'],
                #     'learning_rate':[0.001,0.005,0.01,0.05,0.1,0.2]}
                
                # ebm_parameter_dict = {
                #     'max_bins': [8, 16, 32, 64, 128, 256, 512],
                #     'interactions': [2, 4, 8, 16, 32, 64, 128, 256, 512],
                #     'learning_rate': [np.power(10.0,i) for i in range(-6,2)],
                #     'max_rounds': [1000, 2000, 4000, 8000, 16000],
                #     'min_samples_leaf': [1, 2, 4, 8, 10, 15, 20, 25, 50],
                #     'max_leaves': [1, 2, 4, 8, 10, 15, 20, 25, 50],
                #     'binning': ['quantile', 'uniform', 'quantile_humanized'],
                #     'inner_bags': [1, 2, 4, 8, 16, 32, 64, 128],
                #     'outer_bags': [1, 2, 4, 8, 16, 32, 64, 128]}
                
                ebm_parameter_dict = lambda trial: {
                    'max_bins': trial.suggest_int('max_bins', 3, 256),
                    'validation_size': trial.suggest_float('validation_size', 0.1, 0.3),
                    'outer_bags': trial.suggest_int('outer_bags', 4, 16),
                    'inner_bags': trial.suggest_int('inner_bags', 0, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                    'max_leaves': trial.suggest_int('max_leaves', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
                if not args.tune:
                    ebm_parameter_dict = None
                ebm_score = run_experiment(dataset_name, ebm, ebm_parameter_dict, task, global_seed, n_trials=args.n_trials)
                ebm_end = time.time()
                log[dataset_name]['ebm'] = {'score':ebm_score, 'time':ebm_end-ebm_start}
                print(f"ebm | score: {ebm_score} | time: {ebm_end-ebm_start}")
                save(filename,log)

            if 'ebm_no_interactions' in args.algs:
                ebm_start = time.time()
                if task == 'regression':
                    ebm = ExplainableBoostingRegressor(interactions=0)
                elif task == 'classification':
                    ebm = ExplainableBoostingClassifier(interactions=0)
                # ebm_parameter_dict = {'binning':['uniform','quantile','quantile_humanized'],
                #     'learning_rate':[0.001,0.005,0.01,0.05,0.1,0.2]}
                ebm_parameter_dict = lambda trial: {
                    'max_bins': trial.suggest_int('max_bins', 3, 256),
                    'validation_size': trial.suggest_float('validation_size', 0.1, 0.3),
                    'outer_bags': trial.suggest_int('outer_bags', 4, 16),
                    'inner_bags': trial.suggest_int('inner_bags', 0, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                    'max_leaves': trial.suggest_int('max_leaves', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
                if not args.tune:
                    ebm_parameter_dict = None
                ebm_score = run_experiment(dataset_name, ebm, ebm_parameter_dict, task, global_seed, n_trials=args.n_trials)
                ebm_end = time.time()
                log[dataset_name]['ebm_no_interactions'] = {'score':ebm_score, 'time':ebm_end-ebm_start}
                print(f"ebm_no_interactions | score: {ebm_score} | time: {ebm_end-ebm_start}")
                save(filename,log)
           
            if 'esr' in args.algs:
                esr_start = time.time()
                gp_config = {
                    'population_size':100,
                    'generations':5,
                    'function_set':('add','sub','mul','div','shape'),
                    'verbose':True,
                    'random_state':global_seed,
                    'const_range':None,
                    'n_jobs':1,
                    'p_crossover':0.7,
                    'p_subtree_mutation':0.05,
                    'p_point_mutation':0.1,
                    'p_hoist_mutation':0.05,
                    'p_point_replace':0.2,
                    'parsimony_coefficient':0.0,
                    'metric': ('mse' if task == 'regression' else 'log loss'),
                    'parsimony_coefficient':0.0
                }
                
               
                esr_parameter_dict = {
                    # 'parsimony_coefficient': scipy.stats()
                    'optim_dict': DictionaryDistribution({
                        'alg':['adam'],
                        'lr': scipy.stats.loguniform(1e-5,1),
                        'max_n_epochs':[200],
                        'tol':[1e-4],
                        'n_iter_no_change':[10],
                        'task':[task],
                        'device':['cuda'],
                        'batch_size':[20000],
                        'shape_class':[ShapeNN],
                        'constructor_dict': [{
                            'n_hidden_layers':10,
                            'width':10,
                            'activation_name':'ELU'
                        }]
                    })
                }
                categorical_variables_dict = create_categorical_variable_dict(dataset_name,task)
                print(categorical_variables_dict)
                if task == 'regression':
                    esr = SymbolicRegressor(**gp_config, categorical_variables=categorical_variables_dict)
                elif task == 'classification':
                    esr = SymbolicClassifier(**gp_config, categorical_variables=categorical_variables_dict)

                esr_score = run_experiment(dataset_name, esr, esr_parameter_dict, task, random_state=global_seed)
                esr_end = time.time()
                log[dataset_name]['esr'] = {'score':esr_score, 'time':esr_end-esr_start}
                print(f"esr | score: {esr_score} | time: {esr_end-esr_start}")
                save(filename,log)


if __name__ == '__main__':
    main()
