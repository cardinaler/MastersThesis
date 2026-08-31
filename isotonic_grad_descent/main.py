import numpy as np
from heat_inv_problem import heat_inv_problem

# Обратная задача теплопроводности

# параметры задачи
def test_func(x):
    if x < 0:
        x = -x
    if -1 < x < 1:
        return x + 1
    elif x >= 1:
        return 2
    else:
        return 0


a = 1.0
T_ = [0.01, 0.1, 0.5, 1, 1.5]
alpha = 0
tau = 1         

# сетка
N = 512
L = 12.0

x = np.linspace(0, L, N)

# истинное решение
#u_0 = np.exp(-2 * x**2) + 4 * np.exp(-5 * (x - 2)**2)
#u_0 = 1 / (1 + np.exp(-5 * x))
#u_0 = np.where(x > 0, 1.0, 0.0)
#u_0 = np.maximum(-x + 1, 0)
#u_0 = np.array([test_func(t) for t in x])
u_0 = np.exp(-x * x)


problem = heat_inv_problem(a, x)
for T in T_:
    problem.create_task(T, u_0)
    problem.solve_task(alpha, tau)
    problem.visualize_task([0, 4], [-0.5, 1.5], f"gauss_T={T}")