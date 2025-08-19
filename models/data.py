"""Data loading utilities for image-to-image sea ice tasks.

This module provides a lightweight xarray -> PyTorch dataset pipeline that reads
Zarr or NetCDF files prepared earlier, extracts predictor and target variables,
and returns torch tensors shaped (C, H, W) per time-step or stacked timesteps.
"""
from pathlib import Path
from typing import List

import numpy as np
import xarray as xr
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split


class XarrayImageDataset(Dataset):
    """A simple Dataset that reads a single xarray Dataset and yields samples.

    Parameters
    ----------s
    ds: xarray.Dataset or path
        The dataset containing variables. If path, opened with xarray.open_zarr/open_dataset.
    input_vars: list[str]
        Names of predictor variables in the dataset (e.g., ['u10','v10','tas']).
    target_var: str
        Name of the prediction target (e.g., 'siconc').
    lead: int
        How many timesteps ahead to predict (monthly lead).
    """
    def __init__(self, ds, input_vars: List[str], target_var: str, lead: int = 1):
        if isinstance(ds, (str, Path)):
            p = Path(ds)
            if p.suffix == '.zarr' or p.is_dir():
                self.ds = xr.open_zarr(str(p), consolidated=True)
            else:
                self.ds = xr.open_dataset(str(p))
        else:
            self.ds = ds

        self.input_vars = input_vars
        self.target_var = target_var
        self.lead = int(lead)

        # Expect 'time' dimension
        if 'time' not in self.ds.dims:
            raise ValueError('Dataset must have a time dimension')

        self.n = self.ds.sizes['time'] - self.lead

    def __len__(self):
        return max(0, self.n)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        # arr expected (C, H, W) or (H, W)
        if arr.ndim == 2:
            arr = np.expand_dims(arr, 0)
        return torch.from_numpy(arr.astype('float32'))

    def __getitem__(self, idx: int):
        # inputs: stack input_vars at time idx
        time0 = idx
        inputs = []
        for v in self.input_vars:
            da = self.ds[v].isel(time=time0)
            arr = np.array(da)
            inputs.append(arr)
        inp = np.stack(inputs, axis=0)  # (C, H, W)

        # target is target_var at time idx + lead
        tgt_da = self.ds[self.target_var].isel(time=idx + self.lead)
        tgt = np.array(tgt_da)

        # Convert time to an integer (ns since epoch) so the default collate can
        # stack the values into a batch. Returning numpy.datetime64 objects
        # causes DataLoader to raise TypeError during collation.
        # get raw time value (may be numpy.datetime64 or similar)
        t_ns = None
        t_val = None
        try:
            t_val = self.ds['time'].values[idx]
            # normalize to nanosecond resolution then cast to int64
            t_ns = np.datetime64(t_val).astype('datetime64[ns]').astype('int64')
        except Exception:
            # fallback: try int() then finally string
            try:
                t_ns = int(self.ds['time'].values[idx])
            except Exception:
                t_ns = str(self.ds['time'].values[idx])

        return self._to_tensor(inp), self._to_tensor(tgt), t_ns


def get_dataloaders(ds, input_vars: List[str], target_var: str, lead: int = 1,
                    batch_size: int = 8, test_fraction: float = 0.2, num_workers: int = 4,
                    time_split: bool = False):
    """Create train/test dataloaders from an xarray dataset or filepath.

    Parameters
    ----------
    ds, input_vars, target_var, lead: as in XarrayImageDataset
    batch_size, test_fraction, num_workers: DataLoader params
    time_split: bool
        If True, perform a chronological split (earliest samples -> train,
        latest samples -> test). If False (default), a randomized split is used.

    Use observational/validation data separately as `val_loader` when training.
    Returns (train_loader, test_loader).
    """

    dataset = XarrayImageDataset(ds, input_vars, target_var, lead=lead)
    n = len(dataset)
    if n == 0:
        raise ValueError('Dataset contains no samples')

    # Determine test/train sizes (ensure at least one train sample when possible)
    ntest = int(n * test_fraction)
    if n > 1 and test_fraction > 0 and ntest == 0:
        ntest = 1
    ntrain = n - ntest
    if ntrain <= 0:
        ntest = max(0, n - 1)
        ntrain = n - ntest
        if ntrain <= 0:
            raise ValueError('Not enough samples after splitting; reduce test_fraction or provide more data')

    if time_split:
        # Chronological split: first ntrain indices -> train, last ntest -> test
        indices = list(range(n))
        train_idx = indices[:ntrain]
        test_idx = indices[ntrain: ntrain + ntest]
        train_ds = torch.utils.data.Subset(dataset, train_idx)
        test_ds = torch.utils.data.Subset(dataset, test_idx)
    else:
        # Random split (default)
        train_ds, test_ds = random_split(dataset, [ntrain, ntest])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=not time_split,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
