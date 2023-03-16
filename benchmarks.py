import os
from pmlb import fetch_data
from sklearn.model_selection import KFold, ParameterGrid, ParameterSampler, StratifiedKFold, train_test_split, StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score, r2_score
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
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

import scipy

from load_data import load_data


regression_dataset_names = [
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
    'stress_strain':[]
}
non_pmlb_datasets = {
    'regression':['concrete','servo','yacht','energy_efficiency_1','energy_efficiency_2','boston','california','stress_strain'],
    'classification':[]
}


def score(model, X, y_true, task, raw=False):
    if task == 'regression':
        y_pred = model.predict(X)
        if raw:
            res = mean_squared_error(y_true,y_pred)
        else:
            res = r2_score(y_true,y_pred)
        print(f"{model} | score: {score}")
    elif task == 'classification':
        y_pred_proba = model.predict_proba(X)[:, 1]
        if raw:
            res = log_loss(y_true,y_pred)
        else:
            res = roc_auc_score(y_true,y_pred_proba)
        print(f"{model} | score: {score}")
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



def run_experiment(dataset_name, org_model, parameter_dict, task, random_state, return_model=False):

    df = pd.DataFrame(columns=['dataset','model','fold','score','time','timestamp'])

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    model_name = org_model.__class__.__name__

    file_name = f"{dataset_name}_{model_name}_{timestamp}.csv"

    file_path = os.path.join('results',file_name)


    if not isinstance(org_model, LinearRegression): 
        org_model.set_params(random_state=random_state)
    random_state = check_random_state(random_state)
    dataset = load_df(dataset_name,task)
    X = dataset.drop(columns=['target'])
    y = dataset[['target']]

    # y_max = y.max()
    # y_min = y.min()

    # y = (y-y_min)/(y_max-y_min)

    # First, we choose a validation set for hyper-parameter tuning
    if task == 'classification':
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.09, random_state=random_state)
        train_index, val_index = next(sss.split(X,y))
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    elif task == 'regression':
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.09, random_state=random_state)

    categorical_variables = categorical_variables_per_dataset[dataset_name]
    is_cat = np.array([i in categorical_variables for i in range(len(X.columns))])

    cat_cols = X.columns.values[is_cat]
    num_cols = X.columns.values[~is_cat]

    if isinstance(org_model, SymbolicRegressor) or isinstance(org_model, SymbolicClassifier):
        categories = []
        for i in categorical_variables:
            uniques = list(X.iloc[:,i].unique())
            categories.append(sorted(uniques))
        
        cat_pipe = Pipeline([('ordinal', OrdinalEncoder(categories=categories))])
        num_pipe = Pipeline([('std',StandardScaler())])
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

    if task == 'classification':
        splitter = StratifiedKFold(n_splits=10,shuffle=True,random_state=random_state)
    else:
        splitter = KFold(n_splits=10,shuffle=True,random_state=random_state)

    best_params = None
    cv_scores = []

    for i, (train_index, test_index) in tqdm(enumerate(splitter.split(X_train,y_train))):

        time_start = time.time()
        
        X_cv_train, y_cv_train = X_train.iloc[train_index], y_train.iloc[train_index]
        X_cv_test, y_cv_test = X_train.iloc[test_index], y_train.iloc[test_index]

        X_cv_train = processing.fit_transform(X_cv_train)
        y_cv_train = y_scaler.fit_transform(y_cv_train)[:,0]

        X_cv_test = processing.transform(X_cv_test)
        y_cv_test = y_scaler.transform(y_cv_test)[:,0]

        if (i == 0) and (parameter_dict is not None) : # first split is used for hyper-parameter tuning
            X_cv_val = processing.transform(X_val)
            y_cv_val = y_scaler.transform(y_val)[:,0]

            ps = ParameterSampler(parameter_dict,n_iter=5000,random_state=random_state)

            def operation(p):
                # print(f"Testing {p}")
                model = clone(org_model)
                model.set_params(**p)
                model.fit(X_cv_train,y_cv_train)
                print(f"R2: {score(model,X_cv_val,y_cv_val,task,raw=False)}") 
                return (p, score(model,X_cv_val,y_cv_val,task,raw=True))

            best_params, _ = find_extremum(operation, ps, 'max')
            print(f"Best parameters: {best_params}")
        
        model = clone(org_model)
        
        if best_params is not None:
            model.set_params(**best_params)

        if isinstance(org_model, SymbolicRegressor) or isinstance(org_model, SymbolicClassifier):
            if i == 0:
                model.optim_dict['keep_models'] = True
            else:
                model.optim_dict['keep_models'] = False

        model.fit(X_cv_train, y_cv_train)
        new_score = score(model,X_cv_test,y_cv_test,task)
        cv_scores.append(new_score)
        print(cv_scores)

        time_end = time.time()

        # Check if model has timestamp parameter
        if hasattr(model, 'timestamp'):
            fold_timestamp = model.timestamp
        else:
            fold_timestamp = "Not available"

        # Create a new dataframe row
        new_df_row = pd.DataFrame({
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
        return np.mean(cv_scores), np.std(cv_scores), model
    return np.mean(cv_scores), np.std(cv_scores)

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
    args = parser.parse_args()



    global_seed = 0
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

            if 'lr' in args.algs:
                lr_start = time.time()
                if task == 'regression':
                    lr = LinearRegression()
                elif task == 'classification':
                    lr = LogisticRegression()
                # lr_parameter_dict = {}
                lr_parameter_dict = None
                lr_score = run_experiment(dataset_name, lr, lr_parameter_dict, task, global_seed)
                lr_end = time.time()
                log[dataset_name]['lr'] = {'score':lr_score, 'time':lr_end-lr_start}
                print(f"lr | score: {lr_score} | time: {lr_end-lr_start}")
                save(filename,log)

            if 'xgb' in args.algs:
                xgb_start = time.time()
                if task == 'regression':
                    xgb = XGBRegressor()
                elif task == 'classification':
                    xgb = XGBClassifier()

                # xgb_parameter_dict = {'max_depth': [3, 5, 6, 10, 15, 20],
                #             'subsample': np.arange(0.5, 1.0, 0.1),
                #             'n_estimators' : [100, 250, 500, 1000],
                #             'learning_rate' : [0.001,0.01, 0.05, 0.1, 0.2,0.3],
                #             'gamma' : [0,0.1,0.2,0.3,0.4],
                #             'colsample_bytree': np.arange(0.4, 1.0, 0.1),
                #             'colsample_bylevel': np.arange(0.4, 1.0, 0.1)}
                xgb_parameter_dict = {
                'n_estimators': [1, 2, 4, 8, 10, 20, 50, 100, 200, 250, 500, 1000],
                'max_depth': [2, 5, 10, 20, 25, 50, 100, 2000],
                'learning_rate' : [0.001,0.01, 0.05, 0.1, 0.2,0.3],
                'gamma' : [0,0.1,0.2,0.3,0.4],
                'colsample_bytree': np.arange(0.4, 1.0, 0.1),
                'colsample_bylevel': np.arange(0.4, 1.0, 0.1)}

                xgb_parameter_dict = None
                xgb_score = run_experiment(dataset_name, xgb, xgb_parameter_dict, task, global_seed)
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
                svm_score = run_experiment(dataset_name, svm, svm_parameter_dict, task, global_seed)
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
                
                ebm_parameter_dict = {
                    'max_bins': [8, 16, 32, 64, 128, 256, 512],
                    'interactions': [2, 4, 8, 16, 32, 64, 128, 256, 512],
                    'learning_rate': [np.power(10.0,i) for i in range(-6,2)],
                    'max_rounds': [1000, 2000, 4000, 8000, 16000],
                    'min_samples_leaf': [1, 2, 4, 8, 10, 15, 20, 25, 50],
                    'max_leaves': [1, 2, 4, 8, 10, 15, 20, 25, 50],
                    'binning': ['quantile', 'uniform', 'quantile_humanized'],
                    'inner_bags': [1, 2, 4, 8, 16, 32, 64, 128],
                    'outer_bags': [1, 2, 4, 8, 16, 32, 64, 128]}
                ebm_parameter_dict = None
                ebm_score = run_experiment(dataset_name, ebm, ebm_parameter_dict, task, global_seed)
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
                ebm_parameter_dict = None
                ebm_score = run_experiment(dataset_name, ebm, ebm_parameter_dict, task, global_seed)
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
