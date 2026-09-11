from config import Config
import torch
import torch.nn as nn
from neuralop.models import FNO

def create_model(cfg: Config):
    return FNO(
        n_modes=cfg.n_modes,
        hidden_channels=cfg.hidden_channels,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels
    ).to(cfg.device)


def train_fno(model, train_loader, test_loader, cfg: Config):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.MSELoss()

    print("Старт обучения")

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(cfg.device), batch_y.to(cfg.device)
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            assert out.shape == batch_y.shape, f"Mismatch: {out.shape} vs {batch_y.shape}"
            train_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    out = model(batch_x.to(cfg.device))
                    test_loss += criterion(out, batch_y.to(cfg.device)).item()
            print(f"Epoch {epoch+1:02d}, Train Loss: {train_loss/len(train_loader):.6f}, Test Loss: {test_loss/len(test_loader):.6f}")

    return model
