# all imports
import xarray as xr
import zarr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import random
from math import isqrt, ceil
from typing import List, Tuple, Optional, Union


class GOES16Dataset(Dataset):
    """PyTorch Dataset for GOES-16 satellite data with patching and masking capabilities."""
    
    def __init__(
        self, 
        zarr_path: str,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
        time_indices: Optional[Union[int, List[int]]] = None,
        num_patches: int = 64,
        mask_ratio: float = 0.75,
        patch_size: Optional[Tuple[int, int]] = None,
        transform: Optional[callable] = None,
        return_coords: bool = False
    ):
        """
        Initialize GOES-16 Dataset
        
        Args:
            zarr_path: Path to zarr file
            lat_range: Tuple of (min_lat, max_lat)
            lon_range: Tuple of (min_lon, max_lon)  
            time_indices: Single time index or list of time indices to use
            num_patches: Approximate number of patches per image
            mask_ratio: Fraction of patches to mask
            patch_size: Optional fixed patch size (H, W). If None, uses adaptive patching
            transform: Optional transform to apply to patches
            return_coords: Whether to return coordinate information
        """
        self.zarr_path = zarr_path
        self.lat_range = lat_range
        self.lon_range = lon_range
        self.num_patches = num_patches
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.transform = transform
        self.return_coords = return_coords
        
        # Load and prepare data
        self.dataset = self._load_dataset()
        
        # Handle time indices
        if time_indices is None:
            self.time_indices = list(range(self.dataset.sizes['t']))
        elif isinstance(time_indices, int):
            self.time_indices = [time_indices]
        else:
            self.time_indices = time_indices
            
        # Pre-compute patch grid dimensions
        self._setup_patch_grid()
        
    def _load_dataset(self):
        """Load and prepare the zarr dataset."""
        dataset = xr.open_zarr(self.zarr_path)
        # Adjust longitude coordinates
        dataset = dataset.assign_coords(lon=((dataset.lon + 180) % 360 - 180))
        # Slice to region of interest
        data = dataset["CMI_C13"].sel(
            lat=slice(*self.lat_range), 
            lon=slice(*self.lon_range)
        )
        return data
        
    def _setup_patch_grid(self):
        """Pre-compute patch grid dimensions."""
        # Get spatial dimensions (assuming first time index for sizing)
        sample_data = self.dataset.isel(t=0)
        H, W = sample_data.sizes['lat'], sample_data.sizes['lon']
        
        if self.patch_size is not None:
            # Fixed patch size
            patch_h, patch_w = self.patch_size
            self.grid_rows = H // patch_h
            self.grid_cols = W // patch_w
            self.actual_num_patches = self.grid_rows * self.grid_cols
        else:
            # Adaptive patch size
            best_diff = float('inf')
            for rows in range(1, isqrt(self.num_patches) + 2):
                cols = ceil(self.num_patches / rows)
                if abs(rows * cols - self.num_patches) < best_diff:
                    best_diff = abs(rows * cols - self.num_patches)
                    self.grid_rows, self.grid_cols = rows, cols
            self.actual_num_patches = self.grid_rows * self.grid_cols
            
        # Pre-compute patch boundaries
        self.row_edges = np.linspace(0, H, self.grid_rows + 1, dtype=int)
        self.col_edges = np.linspace(0, W, self.grid_cols + 1, dtype=int)
        
    def __len__(self):
        """Return number of time steps in dataset."""
        return len(self.time_indices)
        
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Returns:
            dict containing:
                - 'visible_patches': Tensor of visible patches [num_patches, H, W]
                - 'target_patches': Tensor of original patches [num_patches, H, W]  
                - 'mask': Tensor indicating which patches are masked [num_patches]
                - 'coords': Optional coordinate information
        """
        time_idx = self.time_indices[idx]
        
        # Load data for this time step
        dataarray = self.dataset.isel(t=time_idx)
        
        # Create patches
        patches = self._create_patches(dataarray)
        
        # Apply masking
        visible_patches, target_patches, mask = self._apply_mask(patches)
        
        # Convert to tensors
        visible_tensor = torch.stack([torch.from_numpy(p.values).float() for p in visible_patches])
        target_tensor = torch.stack([torch.from_numpy(p.values).float() for p in target_patches])
        mask_tensor = torch.from_numpy(mask).long()
        
        # Apply transforms if provided
        if self.transform:
            visible_tensor = self.transform(visible_tensor)
            target_tensor = self.transform(target_tensor)
            
        result = {
            'visible_patches': visible_tensor,
            'target_patches': target_tensor,
            'mask': mask_tensor,
            'time_index': time_idx
        }
        
        # Add coordinate information if requested
        if self.return_coords:
            result['coords'] = {
                'lat': dataarray.lat.values,
                'lon': dataarray.lon.values,
                'time': dataarray.t.values
            }
            
        return result
        
    def _create_patches(self, dataarray):
        """Create patches from the data array."""
        patches = []
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                patch = dataarray.isel(
                    lat=slice(self.row_edges[i], self.row_edges[i+1]),
                    lon=slice(self.col_edges[j], self.col_edges[j+1])
                )
                patches.append(patch)
        return patches
        
    def _apply_mask(self, patches):
        """Apply random masking to patches."""
        num_patches = len(patches)
        num_mask = int(num_patches * self.mask_ratio)
        
        # Randomly sample patches to mask
        all_indices = list(range(num_patches))
        mask_indices = set(random.sample(all_indices, num_mask))
        
        # Create visible patches (masked patches filled with zeros)
        visible_patches = [
            patch if i not in mask_indices else xr.zeros_like(patch)
            for i, patch in enumerate(patches)
        ]
        
        # Keep original patches as targets
        target_patches = patches
        
        # Create mask array
        mask = np.array([1 if i in mask_indices else 0 for i in range(num_patches)])
        
        return visible_patches, target_patches, mask
        
    def get_sample_shape(self):
        """Get the shape of a single patch for model design."""
        sample = self[0]
        return sample['visible_patches'].shape[1:]  # [H, W]
        
    def get_patch_grid_info(self):
        """Get information about the patch grid."""
        return {
            'grid_rows': self.grid_rows,
            'grid_cols': self.grid_cols,
            'num_patches': self.actual_num_patches,
            'row_edges': self.row_edges,
            'col_edges': self.col_edges
        }


# Utility functions for creating data loaders
def create_dataloader(
    zarr_path: str,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    time_indices: Optional[Union[int, List[int]]] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs
) -> DataLoader:
    """
    Create a PyTorch DataLoader for GOES-16 data.
    
    Args:
        zarr_path: Path to zarr file
        lat_range: Tuple of (min_lat, max_lat)
        lon_range: Tuple of (min_lon, max_lon)
        time_indices: Time indices to use
        batch_size: Batch size for data loader
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **dataset_kwargs: Additional arguments for GOES16Dataset
        
    Returns:
        DataLoader instance
    """
    dataset = GOES16Dataset(
        zarr_path=zarr_path,
        lat_range=lat_range,
        lon_range=lon_range,
        time_indices=time_indices,
        **dataset_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


# # Example usage and testing
# if __name__ == "__main__":
#     # Example usage
#     zarr_path = "/notebook_dir/public/mickellals-public/goes-16-2003-10-weeks.tmp.zarr"
    
#     # Create dataset
#     dataset = GOES16Dataset(
#         zarr_path=zarr_path,
#         lat_range=(25.0, 50.0),  # Example lat range
#         lon_range=(-125.0, -65.0),  # Example lon range
#         time_indices=[0, 1, 2, 3, 4],  # First 5 time steps
#         num_patches=64,
#         mask_ratio=0.75
#     )
    
#     print(f"Dataset length: {len(dataset)}")
#     print(f"Sample patch shape: {dataset.get_sample_shape()}")
#     print(f"Patch grid info: {dataset.get_patch_grid_info()}")
    
#     # Get a sample
#     sample = dataset[0]
#     print(f"Visible patches shape: {sample['visible_patches'].shape}")
#     print(f"Target patches shape: {sample['target_patches'].shape}")
#     print(f"Mask shape: {sample['mask'].shape}")
#     print(f"Number of masked patches: {sample['mask'].sum().item()}")
    
#     # Create data loader
#     dataloader = create_dataloader(
#         zarr_path=zarr_path,
#         lat_range=(25.0, 50.0),
#         lon_range=(-125.0, -65.0),
#         time_indices=list(range(10)),
#         batch_size=4,
#         shuffle=True,
#         num_patches=64,
#         mask_ratio=0.75
#     )
    
#     # Test data loader
#     for batch_idx, batch in enumerate(dataloader):
#         print(f"Batch {batch_idx}:")
#         print(f"  Visible patches: {batch['visible_patches'].shape}")
#         print(f"  Target patches: {batch['target_patches'].shape}")
#         print(f"  Masks: {batch['mask'].shape}")
#         if batch_idx == 0:  # Only show first batch
#             break