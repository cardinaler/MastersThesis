import numpy as np
from heat_inv_problem import heat_inv_problem
import matplotlib.pyplot as plt

# Обратная задача теплопроводности

# параметры задачи
def test_func(x):
    if x < 0:
        x = -x
    return np.maximum(-x + 1, 0)

a = 1.0
T_ = [0.01, 0.1, 0.5, 1, 1.5]
alpha = 0
tau = 1         

# сетка
N = 512
L = 12.0

x = np.linspace(-L, L, N)

# истинное решение
u_0 = np.array([test_func(t) for t in x])

problem = heat_inv_problem(a, x)
for T in T_:
    problem.create_task(T, u_0)
    problem.solve_task(alpha, tau)
    problem.visualize_task([-4, 4], [-0.5, 1.5], f"std_line_T={T}")