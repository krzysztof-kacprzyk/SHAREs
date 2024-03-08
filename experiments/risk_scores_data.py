from scipy.interpolate import BSpline, UnivariateSpline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# range 0-50, ylim -1,2.0
def get_nodes_feature():
    mini = 0
    maxi = 50
    k = 3
    n_interior_knots = 5
    internal_knots = np.linspace(mini,maxi,n_interior_knots + 2)[1:-1]
    knots = np.r_[[mini]*(k+1),internal_knots,[maxi]*(k+1)]
    n = len(knots) - k - 1
    c = [0.6,0.5,0.25,0.2,0.32,0.3,0.38,0.4,0.39] # should have length n
    bs = BSpline(knots,c,k)
    def f(x):
        return bs(x)*3.5-1
    return f

# range 45-70, ylim -1,2.0
def get_age_feature():
    x = np.array([45,47.5,50,52,55,58,62,65,67.5,70])
    y = np.array([-1.0,-0.5,0.0,0.15,0.0,-0.05,0.0,-0.25,-0.75,-1.25])
    us = UnivariateSpline(x,y,s=0)
    return us

# range 17-45,ylim -5, 1.0
def get_bmi_feature():
    x = np.array([17,20,25,28,32,35,37,40,45])
    y = np.array([0.1,-0.3,0.1,0.1,0.3,-0.15,-0.5,-1,-1.9])
    us = UnivariateSpline(x,y,s=0)
    return us

# t = np.linspace(0,50,1000)
# nodes_feature = get_nodes_feature()
# plt.plot(t,nodes_feature(t))
# plt.ylim(-1,2.0)
# plt.show()

# t = np.linspace(45,70,1000)
# age_feature = get_age_feature()
# plt.plot(t,age_feature(t))
# plt.ylim(-2.5,0.5)
# plt.show()

# t = np.linspace(17,45,1000)
# bmi_feature = get_bmi_feature()
# plt.plot(t,bmi_feature(t))
# plt.ylim(-5,1.0)
# plt.show()

def generate_data(n_samples, noise=0.0, seed=0, return_X_y=False):
    
    generator = np.random.default_rng(seed)
    
    data = {}
    
    node_range = (0,50)
    age_range = (45,70)
    bmi_range = (17,45)

    f_node = get_nodes_feature()
    f_age = get_age_feature()
    f_bmi = get_bmi_feature()
    
    data['node'] = generator.uniform(node_range[0],node_range[1],n_samples)
    data['age'] = generator.uniform(age_range[0],age_range[1],n_samples)
    data['bmi'] = generator.uniform(bmi_range[0],bmi_range[1],n_samples)

    df = pd.DataFrame(data)
    
    df['target'] = f_node(data['node'])+f_age(data['age'])+f_bmi(data['bmi'])

    df['target'] += generator.normal(0,noise,size=n_samples)
    
    if return_X_y:
        return df.drop(columns=['target']).values, df['target'].values
    else:
        return df