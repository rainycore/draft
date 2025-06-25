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

    """
    dataset = xr.open_zarr(zarr_path)
    # change the coordinate scale to align better
    dataset = dataset.assign_coords(lon=((dataset.lon + 180) % 360 - 180))
    
    # slice to specific lat/lon and time
    data = dataset["CMI_C13"].sel(lat=slice(*lat_range), lon=slice(*lon_range), t=time_index)
    return data


def patchify(array, num_patches=744):
    """ splits the entire dataset into approx number of specified patches 
    (allows for slightly uneven patch sizes if shape doesn't divide evenly)

    parameters:
        array (ndarray): 2D numpy array (H,W)
        num_patches (int): approx number of patches wanted

    returns:
        list of approx num_patches 2D numpy arrays

    """
    H, W = array.shape

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
            top, bottom = row_edges[i], row_edges[i+1]
            left, right = col_edges[j], col_edges[j+1]
            patch = array[top:bottom, left:right]
            patches.append(patch)

    return patches


def apply_mask(patches, mask_ratio=0.15):
    """ 
    """
    num_patches = patches.shape[0]
    num_mask = int(num_patches * mask_ratio)
    
    all_indices = list(range(num_patches))
    mask_indices = set(random.sample(all_indices, num_mask))
    
    visible = np.array([patch if i not in mask_indices else np.zeros_like(patch)
                        for i, patch in enumerate(patches)])
    mask = np.array([1 if i in mask_indices else 0 for i in range(num_patches)])
    
    return visible, mask
