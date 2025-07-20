#!/usr/bin/env python3
"""
Quick debug test to see why we're getting 0 cubes
"""

import sys
from pathlib import Path
sys.path.append('.')

from segmentation_main import segment_vegetation
from base import VegetationSegmentationParameters

def debug_test():
    """Test with very relaxed parameters to see if we can find any cubes"""
    
    # Test with very relaxed parameters
    params = VegetationSegmentationParameters(
        max_spatial_distance=20,      # Increased
        min_vegetation_ndvi=0.2,      # Reduced threshold  
        min_cube_size=5,              # Reduced minimum size
        n_clusters=None               # No limit first
    )
    
    data_file = '../../../data/processed/landsat_multidimensional_Sant_Marti_Ciutat_Vella.nc'
    print('Testing with relaxed parameters...')
    print(f'Parameters: max_distance={params.max_spatial_distance}, min_ndvi={params.min_vegetation_ndvi}, min_size={params.min_cube_size}')
    
    try:
        cubes = segment_vegetation(
            netcdf_path=data_file,
            parameters=params,
            municipality_name='Sant Martí',
            create_visualizations=False,
            output_dir='outputs/debug'
        )
        print(f'Found {len(cubes)} cubes with relaxed parameters')
        
        if len(cubes) > 0:
            print("SUCCESS: Found some cubes!")
            for i, cube in enumerate(cubes[:3]):  # Show first 3
                print(f"  Cube {i}: {cube.area} pixels, heterogeneity={cube.heterogeneity:.3f}")
        else:
            print("No cubes found - there might be an issue with the clustering logic")
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_test()
