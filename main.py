import numpy as np
from gplearn.gplearn.genetic import SymbolicRegressor
from gplearn.gplearn.functions import _function_map
from gplearn.gplearn.fitness import _fitness_map
from gplearn.gplearn.utils import check_random_state
SEED = 0

import torch

import matplotlib.pyplot as plt

np.random.seed(SEED)

from gplearn.gplearn._program import _Program
from gplearn.gplearn.model import ShapeNN

program_config = {
'function_set' : [_function_map['add'],_function_map['sub'],_function_map['mul'],_function_map['div'],_function_map['shape']],
'arities' : {1: [_function_map['shape']], 2: [_function_map['add'],_function_map['sub'],_function_map['mul'],_function_map['div']]},
'init_depth' : (2,5),
'init_method' : 'half and half',
'n_features' : 3,
'const_range' : None,
'metric' : _fitness_map['mse'],
'p_point_replace' : 0.05,
'parsimony_coefficient' : 0.001,
'random_state': check_random_state(SEED),
}

# for i in range(10):
#     program = _Program(**program_config)
#     print(program)
#     print(program.get_possible_subtree_roots({0,1}))

gp_config = {
    'population_size':20,
    'generations':1,
    'function_set':('add','sub','mul','div','shape'),
    'verbose':True,
    'random_state':SEED,
    'const_range':None,
    'n_jobs':1,
    'p_crossover':0.7,
    'p_subtree_mutation':0.05,
    'p_point_mutation':0.1,
    'p_hoist_mutation':0.05,
    'p_point_replace':0.2,
    'parsimony_coefficient':0.0,
    'metric':'mse'
}
optim_dict = {
                'alg':'adam',
                'lr':1e-2,
                'max_n_epochs':400,
                'tol':1e-4,
                'n_iter_no_change':10,
                'task':'regression'
            }
sr = SymbolicRegressor(**gp_config,categorical_variables=[],optim_dict=optim_dict)

# Generate data
x1 = np.linspace(0,1,5)
x2 = np.linspace(0,1,5)
# x3 = np.linspace(0,1,5)
# x4 = np.linspace(1,2,5)
mgrid = np.meshgrid(x1,x2)
z = np.sin(6*mgrid[0] * mgrid[1])
mgrid_stack = np.stack(mgrid,axis=-1)
covariates = mgrid_stack.reshape(-1,2)
target = z.reshape(-1)
target += np.random.randn(target.shape[0]) * 0.01

sr.fit(covariates,target)

print(sr._program)

shape = sr._program.model.shape_functions[0]

t = torch.linspace(0,1,1000)
y_pred = shape(t.unsqueeze(1)).cpu().detach().numpy()
plt.plot(t.cpu().detach().numpy(),y_pred)
plt.show()

# program_list = [_function_map['shape'],_function_map['mul'],0,1]
# program = _Program(**program_config,program=program_list)
# print(program)
# print(program.raw_fitness(covariates,target,np.ones_like(target)))

# shape = program.model.shape_functions[0]
# model_parameters = filter(lambda p: p.requires_grad, shape.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
# print(params)

# t = torch.linspace(0,1,1000)
# y_pred = shape(t.unsqueeze(1)).cpu().detach().numpy()
# plt.plot(t.cpu().detach().numpy(),y_pred)
# plt.show()

# program_list = [_function_map['mul'],0,1]
# program = _Program(**program_config,program=program_list)
# print(program)
# print(program.raw_fitness(covariates,target,np.ones_like(target)))


# nn = ShapeNN(10,10,'ELU')
# t = torch.linspace(0,1,1000)
# y = torch.sin(6*t)

# torch.manual_seed(0)
# optimizer = torch.optim.Adam(nn.parameters(), lr=1e-2)

# def closure():
#     optimizer.zero_grad()
#     pred = nn(t.unsqueeze(1)).flatten()
#     loss_fn = torch.nn.MSELoss()
#     loss = loss_fn(pred.float(), y.float())
#     loss.backward()
#     print(loss.item())
#     return loss

# for i in range(400):
#     optimizer.step(closure)

# y_pred = nn(t.unsqueeze(1))
# plt.plot(t.cpu().detach().numpy(),y_pred.cpu().detach().numpy())
# plt.show()