#!/usr/bin/env python3
"""
Debug test to bypass connectivity enforcement and see raw DBSCAN results
"""

import sys
from pathlib import Path
sys.path.append('.')

# Test raw DBSCAN without connectivity enforcement
import numpy as np
import xarray as xr
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def test_raw_dbscan():
    """Test raw DBSCAN clustering without connectivity enforcement"""
    
    print("Testing raw DBSCAN clustering...")
    
    # Load data
    data_file = '../../../data/processed/landsat_multidimensional_Sant_Marti_Ciutat_Vella.nc'
    data = xr.open_dataset(data_file)
    
    # Create simple valid mask
    valid_mask = ~np.isnan(data['ndvi'].isel(time=0).values)
    
    # Extract coordinates
    valid_coords = np.where(valid_mask)
    pixel_coords = list(zip(valid_coords[0], valid_coords[1]))
    
    print(f"Valid pixels: {len(pixel_coords)}")
    
    # Extract NDVI profiles (vectorized)
    ndvi_data = data['ndvi'].values
    if len(ndvi_data.shape) == 4:  # (time, municipality, y, x)
        ndvi_data = ndvi_data[:, 0, :, :]
    
    y_coords = [coord[0] for coord in pixel_coords]
    x_coords = [coord[1] for coord in pixel_coords]
    ndvi_profiles = ndvi_data[:, y_coords, x_coords].T  # Shape: (n_pixels, n_time)
    
    # Filter for vegetation and valid profiles
    vegetation_mask = np.any(ndvi_profiles >= 0.2, axis=1)
    valid_mask = ~np.any(np.isnan(ndvi_profiles), axis=1)
    final_mask = vegetation_mask & valid_mask
    
    ndvi_profiles = ndvi_profiles[final_mask]
    vegetation_coords = [pixel_coords[i] for i in range(len(pixel_coords)) if final_mask[i]]
    
    print(f"Vegetation pixels: {len(vegetation_coords)}")
    
    # Prepare features for clustering (NDVI + spatial coordinates)
    y_coords_veg = np.array([coord[0] for coord in vegetation_coords])
    x_coords_veg = np.array([coord[1] for coord in vegetation_coords])
    spatial_coords = np.column_stack([y_coords_veg, x_coords_veg])
    
    # Combine NDVI profiles with spatial coordinates
    scaler = StandardScaler()
    ndvi_scaled = scaler.fit_transform(ndvi_profiles)
    
    # Scale spatial coordinates to match max_distance parameter
    max_distance = 20
    spatial_weight = 1.0 / max_distance  # This controls spatial clustering strength
    spatial_scaled = spatial_coords * spatial_weight
    
    # Combine features
    combined_features = np.column_stack([ndvi_scaled, spatial_scaled])
    
    print(f"Combined features shape: {combined_features.shape}")
    
    # Run DBSCAN
    eps = 0.5  # Standard eps value
    min_samples = 5  # Minimum cluster size
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(combined_features)
    
    # Analyze results
    unique_labels = np.unique(cluster_labels)
    valid_labels = unique_labels[unique_labels >= 0]
    n_clusters = len(valid_labels)
    n_noise = np.sum(cluster_labels == -1)
    
    print(f"\nDBSCAN Results:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Noise points: {n_noise}")
    print(f"  Total points: {len(cluster_labels)}")
    
    if n_clusters > 0:
        # Show cluster sizes
        print(f"\nCluster sizes:")
        for label in valid_labels[:10]:  # Show first 10
            cluster_size = np.sum(cluster_labels == label)
            print(f"  Cluster {label}: {cluster_size} pixels")
            
        # Create simple cubes without connectivity enforcement
        cubes_created = 0
        for label in valid_labels:
            cluster_mask = cluster_labels == label
            cluster_size = np.sum(cluster_mask)
            if cluster_size >= 5:  # min_cube_size
                cubes_created += 1
        
        print(f"\nPotential cubes (≥5 pixels): {cubes_created}")
        print(f"SUCCESS: Raw DBSCAN found {cubes_created} valid clusters!")
        
        if cubes_created == 0:
            print("Issue: All clusters are too small (< 5 pixels)")
        
    else:
        print("No clusters found by DBSCAN")
        print("Possible issues:")
        print("  - eps too small")
        print("  - min_samples too large") 
        print("  - Features not properly scaled")

if __name__ == "__main__":
    test_raw_dbscan()
