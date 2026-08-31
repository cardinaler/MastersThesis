import numpy as np
from heat_inv_problem import heat_inv_problem
import matplotlib.pyplot as plt

# Обратная задача теплопроводности

# параметры задачи

def test_func(x):
    if x < 0:
        x = -x
    return np.maximum(0, -x + 1)


a = 1.0
T_ = [0.01, 0.1, 0.5, 1, 1.5]
alpha = 0
tau = 1         

# сетка
N = 512
L = 12.0

x = np.linspace(-L, L, N)

# истинное решение
#u_0 = np.exp(-x**2)
#u_0 = np.exp(-x * x)
#u_0 = np.where(x > 0, 1.0, 0.0)

#u_0 = np.array([test_func(t) for t in x])
u_0 = np.array([test_func(t) for t in x])
#u_0 = np.maximum(-x + 1, 0)


problem = heat_inv_problem(a, x)
plt.figure(figsize=(10, 7))
plt.xlim([-4, 4])
plt.ylim([-0.5, 1.5])
plt.plot(x, u_0, label=f"$u_t(x, 0)$", linewidth=2)
plt.title(f"График $u_t(x, T)$ при различных T")
for T in T_:
    u = problem.create_task(T, u_0)
    plt.plot(x, u, label=f"$u_t(x, {T})$", linewidth=2)


plt.legend()
plt.grid()
plt.savefig(f"line_u_t(x, T).pdf", bbox_inches='tight') # 'bbox_inches="tight"' removes extra whitespace

plt.show()