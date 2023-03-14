import numpy as np
import pandas as pd


def temperature(mass, energy, initial_temp):
    # if initial_temp == 0 then it is assumed to be ice
    # if initial_temp == 100 then it is assumed to be water
    specific_heat_ice = 0.5 # cal/(g*C)
    specific_heat_water = 1.0 # cal/(g*C)
    specific_heat_vapor = 0.48 # cal/(g*C)
    heat_of_fusion = 79.72 # cal/g
    heat_of_vaporization = 540 # cal/g
    
    curr_temp = initial_temp
    energy_left = energy
    if curr_temp < 0: # it's ice
        energy_needed_to_0 = - curr_temp * specific_heat_ice * mass
        if energy_needed_to_0 > energy_left:
            curr_temp += energy_left / (mass * specific_heat_ice)
            return curr_temp
        else:
            curr_temp = 0.0
            energy_left -= energy_needed_to_0
    if curr_temp == 0.0: # it's ice at 0 C
        energy_needed_to_water = heat_of_fusion * mass
        if energy_needed_to_water > energy_left:
            return curr_temp
        else:
            energy_left -= energy_needed_to_water
    if 0.0 <= curr_temp < 100.0: # it's water
        energy_needed_to_100 = (100.0 - curr_temp) * specific_heat_water * mass
        if energy_needed_to_100 > energy_left:
            curr_temp += energy_left / (mass * specific_heat_water)
            return curr_temp
        else:
            curr_temp = 100.0
            energy_left -= energy_needed_to_100
    if curr_temp == 100.0: # it's water at 100 C
        energy_needed_to_vapor = heat_of_vaporization * mass
        if energy_needed_to_vapor > energy_left:
            return curr_temp
        else:
            energy_left -= energy_needed_to_vapor
    if curr_temp >= 100.0: # it's vapor
        curr_temp += energy_left / (mass * specific_heat_vapor)
        return curr_temp
    
    # We should never get here
    return None


def generate_data(n_real_samples, energy_per_mass_range, mass_range, initial_temp_range, seed, noise=0, undersample=False):
    generator = np.random.default_rng(seed)

    if undersample:
        n_samples = n_real_samples * 10
    else:
        n_samples = n_real_samples

    data = {}
    data['energy_per_mass'] = generator.uniform(energy_per_mass_range[0],energy_per_mass_range[1],n_samples)
    data['mass'] = generator.uniform(mass_range[0],mass_range[1],n_samples)
    data['energy'] = data['energy_per_mass'] * data['mass']
    # data['energy'] = generator.uniform(energy_range[0],energy_range[1],n_samples)
    data['initial_temp'] = generator.uniform(initial_temp_range[0],initial_temp_range[1],n_samples)

    # data = {k: generator.uniform(v[0],v[1],n_samples) for k,v in ranges.items()}
    labels = [temperature(m,e,t) for m,e,t in zip(data['mass'],data['energy'],data['initial_temp'])]
    data['temperature'] = labels
    df = pd.DataFrame(data).drop(columns=['energy_per_mass'])
    # df = pd.DataFrame(data)

    if undersample:
        n_100 = sum(df['temperature'] == 100.0)
        to_delete = int(n_100 * 0.8)
        indexes_to_delete = df.index[df['temperature'] == 100.0][:to_delete]
        df = df.drop(index=indexes_to_delete)
        df = df.sample(n_real_samples)

    df['temperature'] += generator.normal(0,noise,size=n_real_samples)
    return df