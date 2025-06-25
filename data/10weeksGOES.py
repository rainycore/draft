# all imports
import xarray as xr
import zarr
import numpy as np
from einops import rearrange
import random
from math import isqrt, ceil

# path to data 
zarr_path = "/notebook_dir/public/mickellals-public/goes-16-2003-10-weeks.tmp.zarr" 


def load_and_slice_zarr(zarr_path, lat_range=(25, 45), lon_range=(-80, -64), time_index='2023-01-01T00:05:06.343806976'):
    """ opening and loading the dataset with specified features

    returns:
        xarray.DataArray of GOES satellite data 

    """
    dataset = xr.open_zarr(zarr_path)
    # change the coordinate scale to align better
    dataset = dataset.assign_coords(lon=((dataset.lon + 180) % 360 - 180))
    
    # slice to specific lat/lon and time
    data = dataset["CMI_C13"].sel(lat=slice(*lat_range), lon=slice(*lon_range), t=time_index)
    return data


def create_patches(dataarray, num_patches=800):
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

    

def apply_mask(patches, mask_ratio=0.15):
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
