# all imports
import xarray as xr
import zarr
import numpy as np
from einops import rearrange
import random
import torch
from torch.utils.data import Dataset
from math import isqrt, ceil

# path to data 
zarr_path = "/notebook_dir/public/mickellals-public/goes-16-2003-10-weeks.tmp.zarr" 

class GOESPatchesDataset(Dataset):
    """
    PyTorch Dataset for GOES-16 Satellite Data that handles creating patches and masking 
    data based on num_patches and mask_ratio
    """
    def __init__(
        self,
        zarr_path: str,
        lat_range: tuple[float, float],
        lon_range: tuple[float, float],
        time_index: str,
        num_patches: int,
        mask_ratio: float
    ):
        """
        Args:
            zarr_path:       path to the .zarr store
            lat_range:       (min_lat, max_lat)
            lon_range:       (min_lon, max_lon)
            time_index:      the time index 
            num_patches:     approx number of patches you want
            mask_ratio:      fraction of patches to mask out
        """
        self.zarr_path   = zarr_path
        self.lat_range   = lat_range
        self.lon_range   = lon_range
        self.num_patches = num_patches
        self.mask_ratio  = mask_ratio

        # only accept and ISO-string for the time index (ex: 2023-01-01T00:05:06.343806976)
        if not isinstance(time_index, str):
            raise ValueError("time_index must be an int or an ISO‐string")
        self.time_index = time_index

        # open the dataset
        ds = xr.open_zarr(self.zarr_path)
        # change the coordinate scale to align better
        ds = ds.assign_coords(lon=((ds.lon + 180) % 360 - 180))
         # slice to specific lat/lon and time with channel #13 (clean longwave)
        self._cube = ds["CMI_C13"].sel(
            lat = slice(*self.lat_range),
            lon = slice(*self.lon_range),
        )

    # only one time frame
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("This dataset only contains a single time index (0).")

        # pick the frame either by position or label
        t = self.time_index
        if isinstance(t, int):
            frame = self._cube.isel(t=t)
        else:
            frame = self._cube.sel(t=t)

        # make patches
        patches = self._create_patches(frame, self.num_patches)

        # mask them
        visible, mask = self._apply_mask(patches, self.mask_ratio)

        # to tensors
        visible_tensor = torch.stack([
            torch.from_numpy(p.values).float()
            for p in visible
        ])  # → [num_patches, H_i, W_i]
        mask_tensor = torch.from_numpy(mask).long()  # → [num_patches]

        return {
            "visible_patches": visible_tensor,
            "mask":            mask_tensor,
            "time_index":      t
        }

    @staticmethod
    def _create_patches(dataarray, num_patches):
        """ splits the entire dataset into approx number of specified patches 
        (allows for slightly uneven patch sizes if shape doesn't divide evenly)

        parameters:
            array (xarray.DataArray): 2D array with dimensions ('lat', 'lon')
            num_patches (int): approx number of patches wanted

        returns:
            list of xarray.DataArray patches (non-overlapping, rectangular)

        """
        H, W = dataarray.sizes['lat'], dataarray.sizes['lon']

        # finding the best grid (rows * cols == num_patches)
        best_diff = float('inf')
        # loop through potential number of rows and compute corresponding cols for it
        for rows in range(1, isqrt(num_patches) + 2):
            cols = ceil(num_patches / rows)
            # track which (rows, cols) would get closest to num_patches
            if abs(rows * cols - num_patches) < best_diff:
                best_diff = abs(rows * cols - num_patches)
                best_rows, best_cols = rows, cols

        # compute patch boundaries (flexible, can be uneven)
        row_edges = np.linspace(0, H, best_rows + 1, dtype=int)
        col_edges = np.linspace(0, W, best_cols + 1, dtype=int)

        # extracts a rectangular slice from original array 
        patches = []
        for i in range(best_rows):
            for j in range(best_cols):
                patch = dataarray.isel(
                    lat=slice(row_edges[i], row_edges[i+1]),
                    lon=slice(col_edges[j], col_edges[j+1])
                )
                patches.append(patch)

        return patches

    

    @staticmethod
    def _apply_mask(patches, mask_ratio):
        """ applies random mask to list of xarray.DataArray patches

        parameters:
            patches: list of xarray.DataArray patches
            mask_ratio: fraction of patches to mask

        returns:
            visible: list of xarray.DataArray patches (masked patches filled with zeros)
            mask: numpy array flagging which patches are visible (0) and masked (1)


        """
        num_patches = len(patches)
        num_mask = int(num_patches * mask_ratio)

        # randomly samples num_mask amount of unique indicies from full index list without replacement
        all_indices = list(range(num_patches))
        mask_indices = set(random.sample(all_indices, num_mask))

        # iterates over all patches and if patch index is in mask_indices, replace it with a zeroed-out patch
        visible = [
            patch if i not in mask_indices else xr.zeros_like(patch)
            for i, patch in enumerate(patches)
        ]

        # iterates over all patches and flags 1 if the patch is masked and 0 if it isn't
        mask = np.array([1 if i in mask_indices else 0 for i in range(num_patches)])
        return visible, mask
