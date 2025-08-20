"""Training loop and utilities for models."""
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm


class Trainer:
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _to_numpy(self, t):
        return t.detach().cpu().numpy()

    def fit(self, train_loader, test_loader=None, val_loader=None, epochs: int = 10, lr: float = 1e-3,
            save_path: Optional[Path] = None, scheduler=None):
        """Train the model. Pass `test_loader` (from model data split) and
        `val_loader` (observational/validation dataset) separately.
        """
        opt = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.L1Loss()

        history = {'train_loss': [], 'test_loss': [], 'val_loss': []}

        for ep in range(epochs):
            self.model.train()
            running = 0.0
            n_valid = 0  # total number of valid (non-NaN) target elements seen
            for xb, yb, _ in tqdm(train_loader, desc=f"Train ep {ep+1}/{epochs}"):
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb)

                # Mask out NaN targets (land/missing values) to avoid NaN losses
                try:
                    mask = ~torch.isnan(yb)
                except Exception:
                    mask = None

                if mask is not None:
                    valid_count = int(mask.sum())
                    # If no valid pixels in batch, skip
                    if valid_count == 0:
                        continue
                    # compute loss only on valid entries
                    loss = loss_fn(pred[mask], yb[mask])
                else:
                    # when mask logic isn't available, treat whole batch as valid
                    valid_count = xb.size(0) * xb.numel() // xb.size(0)  # fallback: per-batch count (best-effort)
                    loss = loss_fn(pred, yb)

                # guard: if loss is NaN skip this batch
                if torch.isnan(loss):
                    continue

                opt.zero_grad()
                loss.backward()
                opt.step()

                # weight by number of valid elements so averages are on a per-pixel basis
                running += float(loss.item()) * valid_count
                n_valid += valid_count

            # compute per-element average; if no valid pixels seen, return NaN to signal issue
            if n_valid == 0:
                train_loss = float('nan')
            else:
                train_loss = running / float(n_valid)

            history['train_loss'].append(train_loss)
            print(f"Epoch {ep+1} train loss: {train_loss:.4f}")

            # evaluate on test split provided by get_dataloaders
            if test_loader is not None:
                test_loss = self.evaluate(test_loader, loss_fn)
                history['test_loss'].append(test_loss)
                print(f"Epoch {ep+1} test loss: {test_loss:.4f}")

            # optional observational validation
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, loss_fn)
                history['val_loss'].append(val_loss)
                print(f"Epoch {ep+1} val loss: {val_loss:.4f}")

            if scheduler is not None:
                scheduler.step()

            if save_path:
                torch.save(self.model.state_dict(), str(save_path))

        return history

    def evaluate(self, loader, loss_fn=nn.L1Loss()):
        self.model.eval()
        running = 0.0
        n = 0  # total number of valid (non-NaN) target elements

        with torch.no_grad():
            for xb, yb, _ in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                pred = self.model(xb)
                mask = ~torch.isnan(yb)
                valid_count = int(mask.sum())
                # skip batches with no valid pixels
                if valid_count == 0:
                    continue
                loss_val = loss_fn(pred[mask], yb[mask]).item()
                # weight by number of valid elements
                running += loss_val * valid_count
                n += valid_count

        if n == 0:
            return float('nan')
        return running / float(n)

    def predict_batch(self, loader, max_batches: Optional[int] = None):
        """Return numpy arrays for predictions and truths (for plotting)."""
        self.model.eval()
        preds = []
        trues = []
        times = []
        with torch.no_grad():
            for i, (xb, yb, t) in enumerate(loader):
                xb = xb.to(self.device)
                out = self.model(xb)
                preds.append(self._to_numpy(out))
                trues.append(self._to_numpy(yb))
                times.append(t)
                if max_batches and i + 1 >= max_batches:
                    break
            if len(preds) == 0:
                raise ValueError('No batches were returned from the loader')
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            # `times` is a list of arrays (one per batch). concatenate if possible
            try:
                times = np.concatenate([np.asarray(t) for t in times], axis=0)
            except Exception:
                # fallback: keep as list
                pass

            # If channel dimension is 1, squeeze to (N, H, W) for convenience
            if preds.shape[1] == 1:
                preds = preds[:, 0, ...]
            if trues.shape[1] == 1:
                trues = trues[:, 0, ...]

            return preds, trues, times
