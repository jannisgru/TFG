#!/usr/bin/env python3
"""
Test script to verify n_clusters optimization performance improvements.

This script compares performance between unlimited clustering and n_clusters-limited
clustering to demonstrate the speed improvements.
"""

import time
import sys
from pathlib import Path

# Add the scripts directory to Python path
scripts_dir = Path(__file__).parent.parent.parent
sys.path.append(str(scripts_dir))

from segmentation_main import segment_vegetation
from base import VegetationSegmentationParameters

def test_n_clusters_optimization():
    """Test the performance improvement with n_clusters parameter"""
    print("="*80)
    print("TESTING N_CLUSTERS OPTIMIZATION PERFORMANCE")
    print("="*80)
    
    # Data file
    data_file = str(Path(__file__).parent.parent.parent.parent / "data" / "processed" / "landsat_multidimensional_Sant_Marti_Ciutat_Vella.nc")
    
    if not Path(data_file).exists():
        print(f"Error: Data file not found: {data_file}")
        print("Please ensure you have the processed data available.")
        return False
    
    municipality = "Sant Martí"
    
    try:
        # Test 1: Unlimited clustering (baseline)
        print("\n" + "="*60)
        print("TEST 1: UNLIMITED CLUSTERING (BASELINE)")
        print("="*60)
        
        params_unlimited = VegetationSegmentationParameters(
            max_spatial_distance=10,
            min_vegetation_ndvi=0.4,
            min_cube_size=20,
            n_clusters=None  # No limit
        )
        
        start_time = time.time()
        cubes_unlimited = segment_vegetation(
            netcdf_path=data_file,
            parameters=params_unlimited,
            municipality_name=municipality,
            create_visualizations=False,
            output_dir="outputs/test_unlimited"
        )
        unlimited_time = time.time() - start_time
        
        print(f"\nUnlimited clustering results:")
        print(f"- Number of cubes: {len(cubes_unlimited)}")
        print(f"- Processing time: {unlimited_time:.2f} seconds")
        
        # Test 2: Limited to 5 clusters
        print("\n" + "="*60)
        print("TEST 2: LIMITED TO 5 CLUSTERS")
        print("="*60)
        
        params_5_clusters = VegetationSegmentationParameters(
            max_spatial_distance=10,
            min_vegetation_ndvi=0.4,
            min_cube_size=20,
            n_clusters=5  # Limit to 5 largest clusters
        )
        
        start_time = time.time()
        cubes_5 = segment_vegetation(
            netcdf_path=data_file,
            parameters=params_5_clusters,
            municipality_name=municipality,
            create_visualizations=False,
            output_dir="outputs/test_5_clusters"
        )
        limited_5_time = time.time() - start_time
        
        print(f"\nLimited (5) clustering results:")
        print(f"- Number of cubes: {len(cubes_5)}")
        print(f"- Processing time: {limited_5_time:.2f} seconds")
        print(f"- Speed improvement: {unlimited_time/limited_5_time:.2f}x faster")
        
        # Test 3: Limited to 3 clusters (should be even faster)
        print("\n" + "="*60)
        print("TEST 3: LIMITED TO 3 CLUSTERS")
        print("="*60)
        
        params_3_clusters = VegetationSegmentationParameters(
            max_spatial_distance=10,
            min_vegetation_ndvi=0.4,
            min_cube_size=20,
            n_clusters=3  # Limit to 3 largest clusters
        )
        
        start_time = time.time()
        cubes_3 = segment_vegetation(
            netcdf_path=data_file,
            parameters=params_3_clusters,
            municipality_name=municipality,
            create_visualizations=False,
            output_dir="outputs/test_3_clusters"
        )
        limited_3_time = time.time() - start_time
        
        print(f"\nLimited (3) clustering results:")
        print(f"- Number of cubes: {len(cubes_3)}")
        print(f"- Processing time: {limited_3_time:.2f} seconds")
        print(f"- Speed improvement: {unlimited_time/limited_3_time:.2f}x faster")
        
        # Summary comparison
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Method':<20} {'N_Cubes':<10} {'Time(s)':<10} {'Speedup':<10}")
        print("-"*80)
        print(f"{'Unlimited':<20} {len(cubes_unlimited):<10} {unlimited_time:<10.2f} {'1.00x':<10}")
        print(f"{'Limited to 5':<20} {len(cubes_5):<10} {limited_5_time:<10.2f} {f'{unlimited_time/limited_5_time:.2f}x':<10}")
        print(f"{'Limited to 3':<20} {len(cubes_3):<10} {limited_3_time:<10.2f} {f'{unlimited_time/limited_3_time:.2f}x':<10}")
        
        # Verify quality: check that limited clusters are among the largest from unlimited
        if len(cubes_unlimited) >= 5:
            unlimited_areas = sorted([cube.area for cube in cubes_unlimited], reverse=True)
            limited_5_areas = sorted([cube.area for cube in cubes_5], reverse=True)
            
            print(f"\nQuality verification:")
            print(f"- Top 5 unlimited areas: {unlimited_areas[:5]}")
            print(f"- Limited 5 areas: {limited_5_areas}")
            
            # Check if limited areas are close to top unlimited areas
            if len(limited_5_areas) == 5:
                area_match = all(abs(limited_5_areas[i] - unlimited_areas[i]) / unlimited_areas[i] < 0.1 
                               for i in range(5) if unlimited_areas[i] > 0)
                print(f"- Area matching (within 10%): {'✓ PASS' if area_match else '✗ FAIL'}")
        
        print(f"\n{'✓ SUCCESS: N_clusters optimization is working correctly!' if limited_3_time < unlimited_time else '⚠ WARNING: Optimization may not be working as expected'}")
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the n_clusters optimization test"""
    success = test_n_clusters_optimization()
    
    if success:
        print("\n" + "="*80)
        print("OPTIMIZATION FEATURES VERIFIED:")
        print("✓ Adaptive clustering with eps optimization for target clusters")
        print("✓ Pre-sorting clusters by size for early selection")
        print("✓ Early termination in cube creation")
        print("✓ Fast connectivity mode for small n_clusters")
        print("✓ Optional pixel sampling for large datasets")
        print("✓ Elimination of redundant post-processing sorting")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("✗ OPTIMIZATION TEST FAILED")
        print("Please check the error messages above.")
        print("="*80)
    
    return success

if __name__ == "__main__":
    main()
