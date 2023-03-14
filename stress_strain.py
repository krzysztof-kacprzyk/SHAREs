import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import pysr

chosen_files = glob.glob("data/stress-strain-curves/T_*_B_1_*.csv")
filename = chosen_files[0]
df = pd.read_csv(filename)
X = df[['Strain']]
y = df['Stress_MPa']

model = pysr.PySRRegressor(
    binary_operators=["+", "*", "/", "-"],
    unary_operators=[
        "log",
        "exp",
        "sin",
    ],
    loss="L2DistLoss()",
    maxsize=10,
    procs=10,
    populations=15,
    niterations=40,
    population_size=33,
    model_selection='accuracy'
)

model.fit(X,y)