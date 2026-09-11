import numpy as np
import torch
from src.heat_operator import heat_operator
from torch.utils.data import TensorDataset, DataLoader
from config import Config

'''
    Генератор обучающей выборки
'''

def generate_dataset(num_samples, num_points, L_max, T_min, T_max, a=1.0):
    """
    Генерирует пары (u0, phi) для обратной задачи теплопроводности.
    Использует БПФ, чтобы описать оператор A_T.
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
            # Функция вида A * exp(-w*x**2) удовлетворяет условиям
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

def prepare_dataloaders(cfg : Config):
    print("Генерация обучающей выборки...")
    u0_data, phi_data, T_data, x_grid = generate_dataset(
        cfg.num_samples, cfg.num_points, cfg.l_max, cfg.t_min, cfg.t_max, cfg.a_coeff
    )

    # Подготовка входа для FNO. Добавляется параметр T как дополнительный канал
    #x_expanded = x_grid.unsqueeze(0).repeat(cfg.num_samples, 1)
    T_expanded = T_data.unsqueeze(1).repeat(1, cfg.num_points) 

    # input_tensor имеет размерность: (Batch, 2 channels, Num_points) 
    # Первый канал это phi(x), а второй T, каждый имеет Num_points значений
    input_tensor = torch.stack([phi_data, T_expanded], dim=1) # (Batch, 2, Num_points)
    target_tensor = u0_data.unsqueeze(1) # Выход: (Batch, Num_points, 1)

    # Разбиение на train/test
    train_size = int((1.0 - cfg.test_split) * cfg.num_samples)
    train_dataset = TensorDataset(input_tensor[:train_size], target_tensor[:train_size])
    test_dataset = TensorDataset(input_tensor[train_size:], target_tensor[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, test_loader, x_grid, input_tensor, target_tensor, train_size