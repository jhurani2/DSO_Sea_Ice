"""Training loop and utilities for models."""
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm


class Trainer:
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fit(self, train_loader, val_loader=None, epochs: int = 10, lr: float = 1e-3,
            save_path: Optional[Path] = None):
        opt = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.L1Loss()

        for ep in range(epochs):
            self.model.train()
            running = 0.0
            for xb, yb, _ in tqdm(train_loader, desc=f"Train ep {ep+1}/{epochs}"):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                running += loss.item()
            print(f"Epoch {ep+1} train loss: {running/len(train_loader):.4f}")

            if val_loader is not None:
                self.model.eval()
                vloss = 0.0
                with torch.no_grad():
                    for xb, yb, _ in val_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        pred = self.model(xb)
                        vloss += loss_fn(pred, yb).item()
                print(f"Epoch {ep+1} val loss: {vloss/len(val_loader):.4f}")

            if save_path:
                torch.save(self.model.state_dict(), str(save_path))

# End Patch
