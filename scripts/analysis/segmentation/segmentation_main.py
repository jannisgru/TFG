"""
Vegetation-focused ST-Cube Segmentation Main Script

This script provides vegetation-specific NDVI clustering segmentation
with local spatial constraints for analyzing vegetation patterns.
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import sys
import warnings
from typing import List, Optional

# Add the segmentation package to path
sys.path.append(str(Path(__file__).parent))

# Import the vegetation-focused components
from base import VegetationSegmentationParameters
from initializers import VegetationNDVIClusteringInitializer
from cube import STCube
from interactive_visualization import InteractiveVisualization

warnings.filterwarnings('ignore')


def segment_vegetation(netcdf_path: str, 
                      parameters: Optional[VegetationSegmentationParameters] = None,
                      municipality_name: str = "Sant Martí",
                      create_visualizations: bool = True,
                      output_dir: str = "outputs/vegetation_clustering") -> List[STCube]:
    """
    Run vegetation-focused NDVI clustering segmentation.
    
    This function performs spatially-aware clustering of vegetation pixels
    based on their NDVI temporal patterns, with local connectivity constraints.
    
    Args:
        netcdf_path: Path to the NetCDF data file
        parameters: VegetationSegmentationParameters object. If None, uses defaults
        municipality_name: Name of the municipality to analyze
        create_visualizations: Whether to create interactive HTML visualizations
        output_dir: Directory to save visualizations
        
    Returns:
        List of STCube objects representing vegetation clusters
    """
    
    print(f"=== Vegetation NDVI Clustering Segmentation ===")
    print(f"Data: {netcdf_path}")
    print(f"Municipality: {municipality_name}")
    
    # Use default parameters if none provided
    if parameters is None:
        parameters = VegetationSegmentationParameters()
        print("Using default vegetation segmentation parameters")
    
    print(f"Parameters: max_distance={parameters.max_spatial_distance}, "
          f"min_vegetation_ndvi={parameters.min_vegetation_ndvi}, "
          f"min_cube_size={parameters.min_cube_size}")
    
    try:
        # Step 1: Load and prepare data
        print("\n1. Loading data...")
        data, valid_mask = load_and_prepare_data(netcdf_path, municipality_name)
        
        if data is None:
            print("Error: Failed to load data")
            return []
        
        # Step 2: Initialize vegetation clustering
        print("\n2. Initializing vegetation NDVI clustering...")
        initializer = VegetationNDVIClusteringInitializer(parameters)
        cubes = initializer.initialize_cubes(data, valid_mask)
        
        if not cubes:
            print("Warning: No vegetation cubes created")
            return []
        
        print(f"Created {len(cubes)} vegetation ST-cubes")
        
        # Step 3: Print summary statistics
        print_vegetation_summary(cubes)
        
        # Step 4: Create visualizations if requested
        if create_visualizations and cubes:
            print(f"\n4. Creating interactive visualizations...")
            create_vegetation_visualizations(cubes, data, output_dir, municipality_name)
        
        print(f"\n=== Vegetation segmentation completed successfully ===")
        return cubes
        
    except Exception as e:
        print(f"Error during vegetation segmentation: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def load_and_prepare_data(netcdf_path: str, municipality_name: str):
    """Load and prepare the NetCDF data for vegetation analysis."""
    
    try:
        # Load the NetCDF file
        data = xr.open_dataset(netcdf_path)
        print(f"Loaded dataset with shape: {dict(data.dims)}")
        
        # Check for required variables
        if 'ndvi' not in data.variables:
            print("Error: NDVI variable not found in dataset")
            return None, None
        
        # Filter by municipality if municipality dimension exists
        if 'municipality' in data.dims:
            if municipality_name in data.municipality.values:
                data = data.sel(municipality=municipality_name)
                print(f"Filtered data for municipality: {municipality_name}")
            else:
                available_munis = list(data.municipality.values)
                print(f"Warning: Municipality '{municipality_name}' not found.")
                print(f"Available municipalities: {available_munis}")
                # Use first available municipality
                if available_munis:
                    municipality_name = available_munis[0]
                    data = data.sel(municipality=municipality_name)
                    print(f"Using municipality: {municipality_name}")
        
        # Create valid mask - pixels with valid NDVI data for all time steps
        ndvi_data = data['ndvi']
        valid_mask = ~np.isnan(ndvi_data).any(dim='time')
        
        n_valid_pixels = valid_mask.sum().item()
        n_total_pixels = valid_mask.size
        
        print(f"Valid pixels: {n_valid_pixels}/{n_total_pixels} "
              f"({100*n_valid_pixels/n_total_pixels:.1f}%)")
        
        if n_valid_pixels < 100:
            print("Warning: Very few valid pixels found")
        
        return data, valid_mask.values
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None


def print_vegetation_summary(cubes: List[STCube]):
    """Print summary statistics for vegetation cubes."""
    
    if not cubes:
        print("No vegetation cubes to summarize")
        return
    
    print(f"\n=== Vegetation Clustering Summary ===")
    print(f"Total vegetation cubes: {len(cubes)}")
    
    # Calculate statistics
    areas = [cube.area for cube in cubes]
    ndvi_means = [np.mean(cube.ndvi_profile) for cube in cubes if hasattr(cube, 'ndvi_profile')]
    ndvi_vars = [cube.temporal_variance for cube in cubes]
    
    print(f"\nArea statistics:")
    print(f"  Mean area: {np.mean(areas):.1f} pixels")
    print(f"  Min area: {np.min(areas):.0f} pixels")
    print(f"  Max area: {np.max(areas):.0f} pixels")
    print(f"  Total area: {np.sum(areas):.0f} pixels")
    
    if ndvi_means:
        print(f"\nNDVI statistics:")
        print(f"  Mean NDVI: {np.mean(ndvi_means):.3f}")
        print(f"  Min NDVI: {np.min(ndvi_means):.3f}")
        print(f"  Max NDVI: {np.max(ndvi_means):.3f}")
        
    if ndvi_vars:
        print(f"\nTemporal variability:")
        print(f"  Mean temporal variance: {np.mean(ndvi_vars):.4f}")
        print(f"  Min temporal variance: {np.min(ndvi_vars):.4f}")
        print(f"  Max temporal variance: {np.max(ndvi_vars):.4f}")


def create_vegetation_visualizations(cubes: List[STCube], 
                                   data: xr.Dataset, 
                                   output_dir: str,
                                   municipality_name: str):
    """Create interactive HTML visualizations for vegetation cubes."""
    
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize visualization engine
        viz = InteractiveVisualization(output_directory=str(output_path))
        
        print(f"  Creating spatial map...")
        viz.create_interactive_spatial_map(
            cubes, 
            f"vegetation_spatial_map_{municipality_name}.html",
            title=f"Vegetation Clusters - {municipality_name}"
        )
        
        print(f"  Creating time series plots...")
        viz.create_interactive_time_series(
            cubes, data,
            f"vegetation_time_series_{municipality_name}.html",
            title=f"Vegetation NDVI Time Series - {municipality_name}"
        )
        
        print(f"  Creating statistics dashboard...")
        viz.create_interactive_statistics_dashboard(
            cubes,
            f"vegetation_statistics_{municipality_name}.html",
            title=f"Vegetation Statistics - {municipality_name}"
        )
        
        print(f"  Creating 3D surface...")
        viz.create_3d_surface_plot(
            cubes, data,
            f"vegetation_3d_surface_{municipality_name}.html",
            title=f"Vegetation 3D Surface - {municipality_name}"
        )
        
        print(f"Visualizations saved to: {output_path}")
        
    except Exception as e:
        print(f"Warning: Failed to create visualizations: {str(e)}")


# Example usage and testing
if __name__ == "__main__":
    
    # Example with default parameters
    print("Testing vegetation segmentation...")
    
    # You can modify these paths and parameters
    data_path = "D:/Uni/TFG/data/processed/landsat_multidimensional_Sant_Marti_Ciutat_Vella.nc"
    municipality = "Sant Martí"
    
    # Create custom parameters
    params = VegetationSegmentationParameters(
        max_spatial_distance=10,      # Local clustering within 10 pixels
        min_vegetation_ndvi=0.4,      # Focus on areas with NDVI ≥ 0.4
        min_cube_size=10              # Minimum 20 pixels per cluster
    )
    
    # Run vegetation segmentation
    vegetation_cubes = segment_vegetation(
        netcdf_path=data_path,
        parameters=params,
        municipality_name=municipality,
        create_visualizations=True,
        output_dir="../../outputs/vegetation_clustering"
    )
    
    print(f"\nCompleted! Found {len(vegetation_cubes)} vegetation clusters.")
