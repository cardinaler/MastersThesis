import torch
from src.FNO.config import Config
from src.FNO.dataset_generator import prepare_dataloaders
from src.FNO.trainer import create_model, train_fno


cfg = Config()
print(f"Используется устройство: {cfg.device}")

print("Генерация обучающей выборки.")
train_loader, test_loader, x_grid, input_tensor, target_tensor, train_size = prepare_dataloaders(
    cfg)

model = create_model(cfg)
print("Старт обучения")
model = train_fno(model, train_loader, test_loader, cfg)

model.eval()
with torch.no_grad():
    sample_idx = 1
    test_input = input_tensor[train_size +
                              sample_idx:train_size + sample_idx + 1].to(cfg.device)
    true_u0 = target_tensor[train_size + sample_idx].numpy()
    pred_u0 = model(test_input).cpu().numpy()[0]
    phi_vis = test_input[0, 0, :].cpu().numpy()
    T_val = test_input[0, 1, 0].item()

plt.figure(figsize=(10, 5))
plt.plot(x_grid.numpy(), true_u0.reshape(-1),
         label='Истинное $u_0(x)$', color='black', linewidth=2)
plt.plot(x_grid.numpy(), pred_u0.reshape(-1), label='Предсказание FNO',
         color='red', linestyle='dashed', linewidth=2)
plt.plot(x_grid.numpy(), phi_vis,
         label=f'Вход $\\varphi(x)$ (T={T_val:.2f})', color='blue', alpha=0.5)
plt.title(f"Восстановление температуры (T = {T_val:.2f})")
plt.xlabel("Координата x")
plt.ylabel("Температура")
plt.legend()
plt.grid(True)
plt.show()
