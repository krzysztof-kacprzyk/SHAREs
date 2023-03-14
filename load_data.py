import pandas as pd
import glob

def load_concrete():
    # regression, all real
    df = pd.read_csv('data/concrete/Concrete_Data.csv')
    column_names = list(df.columns)
    column_names[-1] = 'target'
    df.columns = column_names
    return df

def load_servo():
    # regression, first two columns are categorical
    df = pd.read_csv('data/servo/servo.data',names=['motor','screw','pgain','vgain','target'])
    return df

def load_yacht():
    # all real, regression
    org_filename = 'data/yacht/yacht_hydrodynamics.data'
    new_lines = []
    with open(org_filename,'r') as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()
        elements = line.split("  ")
        assert (len(elements) == 1) or (len(elements) == 2)
        if len(elements) == 2:
            new_line = elements[0] + " " + elements[1]
        elif len(elements) == 1:
            new_line = line
        elif len(elements) > 2:
            raise ValueError()
        new_lines.append(new_line + "\n")
        
    new_filename = 'data/yacht/yacht_hydrodynamics_new.data'
    with open(new_filename,'w') as file:
        file.writelines(new_lines)
        
    column_names = ['Longitudinal position of the center of buoyancy', 'Prismatic coefficient','Length-displacement ratio','Beam-draught ratio','Length-beam ratio','Froude number','target']
    df = pd.read_csv(new_filename,sep=' ',names=column_names)
    return df

def load_energy_efficiency(y):
    # regression, 5 and 7 are categorical (X6, X8)
    df = pd.read_csv('data/energy_efficiency/ENB2012_data.csv')
    if y == 1:
        df = df.drop(columns=['Y2'])
        column_names = list(df.columns)
        column_names[-1] = 'target'
        df.columns = column_names
    elif y == 2:
        df = df.drop(columns=['Y1'])
        column_names = list(df.columns)
        column_names[-1] = 'target'
        df.columns = column_names
    return df

def load_boston():
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'target']
    df = pd.read_csv('data/boston/boston.csv',header = None, delimiter = r"\s+", names=column_names)
    return df

def load_california():
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing(as_frame=True)
    df = data['data']
    df['target'] = data['target']
    return df

def load_stress_strain(index):
    chosen_files = glob.glob("data/stress-strain-curves/T_*_B_1_*.csv")
    filename = chosen_files[index]
    df = pd.read_csv(filename)
    df.columns = ['Strain', 'target']
    last_index = df.index[-1]
    df.drop(index=last_index,inplace=True)
    return df




def load_data(name):
    if name == 'concrete':
        return load_concrete()
    elif name == 'servo':
        return load_servo()
    elif name == 'yacht':
        return load_yacht()
    elif name == 'energy_efficiency_1':
        return load_energy_efficiency(1)
    elif name == 'energy_efficiency_2':
        return load_energy_efficiency(2)
    elif name == 'boston':
        return load_boston()
    elif name == 'california':
        return load_california()
    elif name == 'stress_strain':
        return load_stress_strain(5)

    
    

    