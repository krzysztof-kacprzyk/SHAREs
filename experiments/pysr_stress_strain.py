import sys
sys.path.append('../')
import numpy as np
from experiments.load_data import load_data
from sklearn.model_selection import train_test_split
import time
import pysr

if __name__ == "__main__":

    # start timer
    start = time.time()

    global_seed = 42
    def get_model(length):
        parameter_dict = {
            'binary_operators':["+", "*", "/", "-"],
            'unary_operators':[
                "log",
                "exp",
                "cos",
            ],
            'loss':"L2DistLoss()",
            'maxsize':length,
            'procs':0,
            'multithreading':False,
            'populations':15,
            'niterations':400,
            'population_size':33,
            'model_selection':'accuracy',
            'random_state':global_seed,
            'deterministic':True,
            'equation_file': 'results/Table_1_results.csv'
        }
        return pysr.PySRRegressor(**parameter_dict)

    dataset_name = 'stress_strain'

    data_dir = 'data/'

    df = load_data('stress_strain', data_dir=data_dir)

    X = df[['Strain']]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    length = 25

    model = get_model(length)
    model.fit(X_train,y_train)

    # end timer
    end = time.time()

    # Save the time to a file
    with open('results/Table_1_time.txt', 'w') as f:
        f.write(str(end - start))