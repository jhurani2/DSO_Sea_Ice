# models

This folder contains a lightweight ML framework for image-to-image sea ice prediction.

Files:
- `data.py` — xarray -> PyTorch dataset and dataloader helpers
- `models.py` — persistence, linear, CNN, ResNet models (PyTorch)
- `train.py` — Trainer class for training loops
- `utils.py` — metrics and checkpoint helpers

Quick usage example

```python
from models import get_dataloaders, SimpleCNN, Trainer

train_loader, val_loader = get_dataloaders('data/cesm2_combined.zarr', input_vars=['u10','v10','tas'], target_var='siconc')
model = SimpleCNN(in_ch=3)
trainer = Trainer(model)
trainer.fit(train_loader, val_loader, epochs=10, lr=1e-3, save_path='models/ckpt.pth')
```

Notes
- This is a minimal starting point. You should add data normalization, advanced augmentations, better schedulers, and logging (TensorBoard/MLflow) for production use.
