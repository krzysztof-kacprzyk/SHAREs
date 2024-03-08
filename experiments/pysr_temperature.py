import sys
sys.path.append('../')
import numpy as np
import time
import pysr
from experiments.temperature import generate_data

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
            'populations':30,
            'niterations':400,
            'population_size':50,
            'model_selection':'accuracy',
            'random_state':global_seed,
            'deterministic':True,
            'equation_file': 'results/Table_3_results.csv'
        }
        return pysr.PySRRegressor(**parameter_dict)

    # Generate data
    df = generate_data(1000, (1,800), (1.0,4.0), (-100,0), seed=0, noise=0.0)

    feature_columns = ['energy','mass','initial_temp']
    target_column = 'temperature'

    X = df[feature_columns].values
    y = df[target_column].values

    df_test = generate_data(1000, (1,800), (1.0,4.0), (-100,0), seed=1, noise=0.0)
    X_test = df_test[feature_columns].values
    y_test = df_test[target_column].values
    length = 40

    model = get_model(length)
    model.fit(X,y)

    # end timer
    end = time.time()

    # Save the time to a file
    with open('results/Table_3_time.txt', 'w') as f:
        f.write(str(end - start))