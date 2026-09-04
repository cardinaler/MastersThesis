import numpy as np
from src.heat_operator import heat_operator
'''
    Генератор обучающей выборки
'''

def generate_dataset(num_samples, num_points, L_max, T_min, T_max, a=1.0):
    """
    Генерирует пары (u0, phi) для обратной задачи теплопроводности.
    Использует БПФ для точного применения оператора A_T.
    """
    # Пространственная сетка [-L_max, L_max)
    x = np.linspace(-L_max, L_max, num_points, endpoint=False)
    dx = x[1] - x[0]

    # Волновые числа для БПФ
    k = np.fft.rfftfreq(num_points, d=dx) * 2 * np.pi

    u0_list = []
    phi_list = []
    T_list = []

    for _ in range(num_samples):
        # Генерация u0(x)
        
        u0 = np.full(num_points, 0)

        # Пока u0 это 1 функция вида гауссианы
        num_components = 1
        for _ in range(num_components):
            A = np.random.uniform(1, 5)
            w = np.random.uniform(0.1, 1.0)
            # Функция вида A * exp(-w*x**2) удовлетворяет условиям:
            u0 += A * np.exp(-w * x**2)

        # Генерация случайного финального момента времени T
        T = np.random.uniform(T_min, T_max)

        phi = heat_operator(a, T, x)(u0)

        u0_list.append(u0)
        phi_list.append(phi)
        T_list.append(T)

    return torch.tensor(np.array(u0_list), dtype=torch.float32), \
           torch.tensor(np.array(phi_list), dtype=torch.float32), \
           torch.tensor(np.array(T_list), dtype=torch.float32), \
           torch.tensor(x, dtype=torch.float32)
