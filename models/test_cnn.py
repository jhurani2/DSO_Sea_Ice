import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from models.data import get_dataloaders
from models.models import SimpleCNN
from models.train import Trainer
from models.utils import mae, rmse

# Make repo importable
repo_root = str(Path(__file__).resolve().parents[1])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


def run_quick_test(ds_path: str, batch_size: int = 8, epochs: int = 2):
    print('Using dataset:', ds_path)
    input_vars = ['siconc']
    target_var = 'siconc'

    train_loader, test_loader = get_dataloaders(ds_path, input_vars, target_var,
                                                lead=1, batch_size=batch_size,
                                                test_fraction=0.2, num_workers=0,
                                                time_split=True, normalize=True, fill_na=0.0, add_mask=True)

    # Build a small CNN — channel count depends on dataset plus optional mask
    sample_batch = next(iter(train_loader))
    xb, yb, t = sample_batch
    in_ch = xb.shape[1]
    print(f'Sample batch shapes xb={xb.shape}, yb={yb.shape}, inferred in_ch={in_ch}')

    model = SimpleCNN(in_ch=in_ch, out_ch=1, features=32)
    trainer = Trainer(model)

    # Quick training loop (very short) to confirm pipeline works
    history = trainer.fit(train_loader, test_loader=test_loader, epochs=epochs, lr=1e-3)
    print('History:', history)

    # Evaluate and predict a few batches
    test_loss = trainer.evaluate(test_loader)
    print('Test L1 loss:', test_loss)

    preds, trues, times = trainer.predict_batch(test_loader, max_batches=5)
    print('Preds shape:', preds.shape, 'Trues shape:', trues.shape)

    print('MAE:', mae(preds, trues), 'RMSE:', rmse(preds, trues))

    # Show first example
    n_show = min(3, preds.shape[0])
    for i in range(n_show):
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        vmin = np.nanmin([trues[i], preds[i]])
        vmax = np.nanmax([trues[i], preds[i]])
        ax[0].imshow(trues[i], origin='lower', vmin=vmin, vmax=vmax, cmap='Blues_r')
        ax[0].set_title('True')
        ax[1].imshow(preds[i], origin='lower', vmin=vmin, vmax=vmax, cmap='Blues_r')
        ax[1].set_title('Pred')
        diff = preds[i] - trues[i]
        ax[2].imshow(diff, origin='lower', cmap='RdBu_r', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
        ax[2].set_title('Diff')
        plt.show()


if __name__ == '__main__':
    # Change this path to the dataset you want to test
    ds_path = '/Users/amanjhurani/LocalData_SeaIce/DSO_Sea_Ice/data/CESM2_Seaiceconc/siconc_SImon_CESM2_historical_r10i1p1f1_gn_190001-194912.nc'
    run_quick_test(ds_path, batch_size=8, epochs=2)
