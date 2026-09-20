import numpy as np
from src.grad_method.heat_inv_problem import heat_inv_problem
import matplotlib.pyplot as plt

# Обратная задача теплопроводности

# параметры задачи
def test_func(x):

    return  8 * np.exp(-5 *(np.abs(x) - 6)**2)
 

a = 1.0
T_ = [2.5]
alpha = 0
tau = 2         

# сетка
N = 256
L = 10.0

x = np.linspace(-L, L, N)

# истинное решение
u_0 = test_func(x)

problem = heat_inv_problem(a, x)
for T in T_:
    problem.create_task(T, u_0)
    problem.solve_task(alpha, tau, 'non_negative')
    problem.visualize_task([0, 10], [0.0, 3.0], f"grad_experiment_fx2")