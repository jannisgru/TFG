#!/usr/bin/env python3
"""
Debug test with more detailed output to see what's happening in connectivity enforcement
"""

import sys
from pathlib import Path
sys.path.append('.')

from segmentation_main import segment_vegetation
from base import VegetationSegmentationParameters

def debug_connectivity_test():
    """Test with debug output to see what's happening in connectivity enforcement"""
    
    # Test with very relaxed parameters and force fast mode
    params = VegetationSegmentationParameters(
        max_spatial_distance=25,      # Even more relaxed
        min_vegetation_ndvi=0.2,      # Reduced threshold  
        min_cube_size=5,              # Reduced minimum size
        n_clusters=10                 # Force fast mode with reasonable target
    )
    
    data_file = '../../../data/processed/landsat_multidimensional_Sant_Marti_Ciutat_Vella.nc'
    print('Testing with relaxed parameters and fast mode...')
    print(f'Parameters: max_distance={params.max_spatial_distance}, min_ndvi={params.min_vegetation_ndvi}, min_size={params.min_cube_size}, n_clusters={params.n_clusters}')
    
    try:
        cubes = segment_vegetation(
            netcdf_path=data_file,
            parameters=params,
            municipality_name='Sant Martí',
            create_visualizations=False,
            output_dir='outputs/debug_fast'
        )
        print(f'Found {len(cubes)} cubes with relaxed parameters (fast mode)')
        
        if len(cubes) > 0:
            print("SUCCESS: Found some cubes with fast mode!")
            for i, cube in enumerate(cubes[:3]):  # Show first 3
                print(f"  Cube {i}: {cube.area} pixels, heterogeneity={cube.heterogeneity:.3f}")
        else:
            print("No cubes found even with fast mode - need to debug further")
            
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_connectivity_test()
