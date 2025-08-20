"""Utility functions: metrics and checkpoint helpers."""
from pathlib import Path
from typing import Tuple

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable


def mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.nanmean(np.abs(pred - truth)))


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((pred - truth) ** 2)))


def r2_score(pred: np.ndarray, truth: np.ndarray) -> float:
    # simple R2
    ss_res = np.sum((truth - pred) ** 2)
    ss_tot = np.sum((truth - np.nanmean(truth)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-12))


def save_checkpoint(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch
    torch.save(model.state_dict(), str(path))


def load_checkpoint(model, path: Path):
    import torch
    state = torch.load(str(path), map_location='cpu')
    model.load_state_dict(state)
    return model


def get_arctic_row_slice(ds_obj, lat_names=('lat', 'latitude', 'y'), threshold=60):
    """Return (start_row, end_row) to crop arrays to Arctic (lat >= threshold).
    Best-effort: looks for common latitude coord names in `ds_obj.coords`.
    If not found or shape mismatch, returns (0, None) meaning no crop.
    """
    if ds_obj is None:
        return 0, None
    for name in lat_names:
        if name in getattr(ds_obj, 'coords', {}):
            lats = np.asarray(ds_obj.coords[name])
            break
    else:
        return 0, None

    if lats.ndim != 1:
        try:
            lats = lats.ravel()
        except Exception:
            return 0, None

    # find indices where latitude meets threshold (handle ascending/descending)
    if lats[0] < lats[-1]:
        idx = np.where(lats >= threshold)[0]
    else:
        idx = np.where(lats >= threshold)[0]

    if idx.size == 0:
        return 0, None
    return int(idx[0]), None


def overlay_mask_image(ax, mask):
    """Overlay a light-gray mask where mask==True (land/missing).
    `mask` should be a boolean 2D array with True on land.
    """
    overlay = np.zeros(mask.shape + (4,), dtype=float)
    overlay[mask, :] = np.array([0.85, 0.85, 0.85, 1.0])
    overlay[~mask, 3] = 0.0
    ax.imshow(overlay, origin='lower', interpolation='nearest')


def plot_epoch_comparison(true_img, pred_img, pers_img, mask=None, ds=None, vmin=0.0, vmax=1.0):
    """Plot True | Persistence | Prediction | Difference with shared colorbar and mask overlay.
    `true_img`, `pred_img`, `pers_img` are 2D numpy arrays. `mask` is boolean True where land.
    If `ds` provided, function will attempt to crop to Arctic using dataset coords.
    """
    start, end = get_arctic_row_slice(ds) if ds is not None else (0, None)
    t = true_img[start:end, :]
    p = pred_img[start:end, :]
    per = pers_img[start:end, :]
    m = mask[start:end, :] if mask is not None else np.zeros_like(t, dtype=bool)

    dif = p - t
    max_abs = max(0.1, float(np.nanmax(np.abs(dif))))

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    ax_t, ax_per, ax_pr, ax_d = axs
    im_t = ax_t.imshow(t, cmap='Blues_r', vmin=vmin, vmax=vmax, origin='lower')
    ax_t.set_title('True')
    ax_t.axis('off')
    overlay_mask_image(ax_t, m)

    im_per = ax_per.imshow(per, cmap='Blues_r', vmin=vmin, vmax=vmax, origin='lower')
    ax_per.set_title('Persistence')
    ax_per.axis('off')
    overlay_mask_image(ax_per, m)

    im_pr = ax_pr.imshow(p, cmap='Blues_r', vmin=vmin, vmax=vmax, origin='lower')
    ax_pr.set_title('Prediction')
    ax_pr.axis('off')
    overlay_mask_image(ax_pr, m)

    im_d = ax_d.imshow(dif, cmap='RdBu_r', vmin=-max_abs, vmax=max_abs, origin='lower')
    ax_d.set_title('Prediction - True')
    ax_d.axis('off')
    overlay_mask_image(ax_d, m)

    # shared colorbar for the first three (SIC)
    sm = ScalarMappable(cmap='Blues_r')
    sm.set_clim(vmin, vmax)
    fig.colorbar(sm, ax=axs[:3].tolist(), orientation='vertical', fraction=0.02, pad=0.02, label='SIC')

    # colorbar for diff
    sm2 = ScalarMappable(cmap='RdBu_r')
    sm2.set_clim(-max_abs, max_abs)
    fig.colorbar(sm2, ax=ax_d, orientation='vertical', fraction=0.03, pad=0.02, label='Diff')

    plt.tight_layout()
    # return images too for callers that may want to animate/update
    return fig, axs, (im_t, im_per, im_pr, im_d)
