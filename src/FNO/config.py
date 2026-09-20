from dataclasses import dataclass
import torch

@dataclass
class Config:
    # Параметры генерации данных
    num_samples: int = 5000
    num_points: int = 256
    l_max: float = 10.0
    t_min: float = 0.1
    t_max: float = 2.0
    a_coeff: float = 1.0

    # Обучение
    batch_size: int = 64
    test_split: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50

    # FNO 
    n_modes: tuple = (20,)
    hidden_channels: int = 64
    in_channels: int = 3   # phi(x), x, T
    out_channels: int = 1  # u0(x)

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')