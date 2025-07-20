# Vegetation-focused ST-Cube Segmentation

A streamlined implementation of spatiotemporal cube segmentation focused specifically on vegetation analysis using NDVI clustering with local spatial constraints.

## Key Features

- **Vegetation-focused NDVI Clustering**: Groups nearby pixels with similar NDVI temporal patterns, focusing only on vegetation areas (NDVI ≥ 0.4)
- **Local Spatial Constraints**: Enforces maximum spatial distance (e.g., 10 pixels) for clustering to capture local vegetation patterns
- **Connected Component Analysis**: Ensures spatial connectivity within clusters using 8-connectivity
- **Interactive HTML Visualizations**: Plotly-based interactive maps, time series plots, and 3D surfaces
- **Simplified Architecture**: Clean, focused design for vegetation analysis
- **Performance Optimizations**: Efficient processing when `n_clusters` parameter is specified

## Quick Start

### Vegetation NDVI Clustering Segmentation

```python
from segmentation_main import segment_vegetation
from base import VegetationSegmentationParameters

# Basic usage with default parameters
vegetation_cubes = segment_vegetation(
    netcdf_path="data/landsat_multidimensional_Sant_Marti.nc",
    municipality_name="Sant Martí"
)

# With custom parameters including performance optimization
params = VegetationSegmentationParameters(
    max_spatial_distance=10,      # Local clustering within 10 pixels
    min_vegetation_ndvi=0.4,      # Focus on areas with NDVI ≥ 0.4
    min_cube_size=20,             # Minimum 20 pixels per cluster
    n_clusters=5                  # Limit to 5 largest clusters (faster processing)
)

vegetation_cubes = segment_vegetation(
    netcdf_path="data/landsat_multidimensional_Sant_Marti.nc",
    parameters=params,
    municipality_name="Sant Martí",
    create_visualizations=True,
    output_dir="outputs/vegetation_clustering"
```

## Performance Optimization

When `n_clusters` is specified, the algorithm applies several optimizations:

1. **Adaptive Clustering**: Automatically adjusts clustering parameters to target the desired number of clusters
2. **Early Termination**: Stops processing once the target number of clusters is reached
3. **Pre-sorting**: Prioritizes larger clusters during creation to ensure quality results
4. **Fast Connectivity Mode**: Uses simplified connectivity checks for small target numbers
5. **Pixel Sampling**: For large datasets with small `n_clusters`, processes a representative sample

**Performance gains**: 2-5x faster processing when `n_clusters ≤ 10` on typical datasets.

## Output

The segmentation produces:

1. **List of STCube objects**: Each representing a vegetation cluster with properties:
   - `pixels`: List of (y, x) coordinates
   - `ndvi_profile`: Mean NDVI time series for the cluster
   - `temporal_variance`: Measure of temporal variability
   - `area`: Number of pixels in the cluster
   - `compactness`: Shape compactness measure

2. **Interactive HTML Visualizations** (if enabled):
   - **Spatial Map**: Interactive map showing cluster locations and colors
   - **Time Series Plot**: NDVI temporal patterns for each cluster
   - **Statistics Dashboard**: Cluster statistics and distributions
   - **3D Surface**: 3D visualization of NDVI patterns

## Parameters

### VegetationSegmentationParameters

- `max_spatial_distance` (default: 10): Maximum distance in pixels for spatial connectivity
- `min_vegetation_ndvi` (default: 0.4): Minimum NDVI value to consider as vegetation
- `min_cube_size` (default: 20): Minimum number of pixels required for a valid cluster
- `n_clusters` (default: None): Maximum number of clusters to return. When specified, enables performance optimizations including adaptive clustering, early termination, and fast connectivity checks. Recommended for large datasets or when only the largest clusters are needed.

## How It Works

1. **Data Loading**: Loads NetCDF file and filters for specified municipality
2. **Vegetation Filtering**: Identifies pixels with NDVI ≥ threshold at any time point
3. **Optimization Check**: If `n_clusters` is specified, applies performance optimizations
4. **Temporal Clustering**: Uses DBSCAN to group pixels with similar NDVI patterns (with adaptive parameters if optimizing)
5. **Spatial Constraints**: Enforces maximum distance constraints between cluster pixels
6. **Connectivity Analysis**: Ensures clusters are spatially connected (fast mode for small `n_clusters`)
7. **Cube Creation**: Converts valid clusters into STCube objects with early termination when target is reached

## Architecture

The package consists of the following core modules:

- `base.py`: VegetationSegmentationParameters configuration class
- `segmentation_main.py`: Main entry point with `segment_vegetation()` function
- `initializers/ndvi_cluster_initializer.py`: VegetationNDVIClusteringInitializer implementation
- `cube.py`: STCube data structure for representing spatiotemporal segments
- `interactive_visualization.py`: HTML visualization generation using Plotly

## Installation Requirements

```bash
pip install numpy xarray pandas plotly scikit-learn scipy
```

## Data Format

The package expects NetCDF files with:
- `ndvi` variable: NDVI values with dimensions (time, y, x) or (time, municipality, y, x)
- `time` dimension: Temporal axis
- `y`, `x` dimensions: Spatial coordinates
- Optional `municipality` dimension for multi-municipality datasets

## Example Output

```
=== Vegetation NDVI Clustering Segmentation ===
Data: data/landsat_multidimensional_Sant_Marti.nc
Municipality: Sant Martí
Parameters: max_distance=10, min_vegetation_ndvi=0.4, min_cube_size=20

1. Loading data...
Loaded dataset with shape: {'time': 139, 'y': 50, 'x': 80}
Valid pixels: 2847/4000 (71.2%)

2. Initializing vegetation NDVI clustering...
Extracting NDVI profiles for vegetation pixels...
Found 1523 vegetation pixels with NDVI ≥ 0.4
Performing spatial clustering with max distance 10...
Found 12 clusters with 234 noise points
Enforcing spatial connectivity and distance constraints...
After connectivity enforcement: 8 spatially connected clusters
Created 8 vegetation ST-cubes

=== Vegetation Clustering Summary ===
Total vegetation cubes: 8
Area statistics:
  Mean area: 127.6 pixels
  Min area: 23 pixels
  Max area: 456 pixels
  Total area: 1021 pixels

NDVI statistics:
  Mean NDVI: 0.562
  Min NDVI: 0.445
  Max NDVI: 0.731
```

# Compare all three approaches
results = run_all_methods(
    netcdf_path="data/landsat_multidimensional_ALL_AMB_municipalities.nc",
    municipality_name="Sant Martí"
)

# Access individual results
standard_results = results['standard']
ndvi_results = results['ndvi_clustering']  
custom_results = results['custom']
```

## Return Values

Each segmentation function returns a tuple of:
- `segmenter`: The STCubeSegmenter instance used
- `cubes`: CubeCollection containing the generated ST-cubes
- `analysis`: Dictionary with quality metrics and statistics

The analysis dictionary contains:
- `stgs`: Spatiotemporal Global Score (quality metric)
- `n_cubes`: Number of generated cubes
- `cube_sizes`: Statistics about cube sizes
- `temporal_extents`: Statistics about temporal coverage
- `quality`: Overall quality metrics

## Interactive Visualizations

The package includes specialized interactive HTML visualizations using Plotly:

```python
# Run interactive visualization demo
from interactive_visualization import run_interactive_visualization_demo

segmenter, cubes, analysis = run_interactive_visualization_demo()
```

This creates:
- **Interactive Spatial Map**: Shows vegetation cluster boundaries and NDVI patterns with hover information
- **NDVI Time Series Plot**: Interactive plot of NDVI evolution for each vegetation cluster
- **Statistics Dashboard**: Multi-panel dashboard with vegetation cluster statistics and distributions  
- **3D NDVI Surface**: 3D visualization of NDVI evolution over space and time

All visualizations are saved as HTML files in the specified output directory.

## Testing

Test the vegetation-focused clustering:

```python
# Run comprehensive test
python test_vegetation_clustering.py
```

This script tests different parameter configurations and verifies the vegetation filtering and spatial constraints.

## Files

The streamlined package contains only the essential files:

- `segmentation_main.py`: Main script with `segment_vegetation()` function
- `base.py`: VegetationSegmentationParameters configuration class
- `cube.py`: STCube data structure for spatiotemporal segments
- `interactive_visualization.py`: Interactive HTML visualizations using Plotly
- `initializers/ndvi_cluster_initializer.py`: VegetationNDVIClusteringInitializer
- `test_vegetation_clustering.py`: Test script for vegetation clustering
- `README.md`: This documentation

## License

This project is part of a university thesis focusing on vegetation analysis using satellite time series data.
- `heterogeneity.py`: Heterogeneity calculation strategies
- `merger.py`: Cube merging policies
- `analysis.py`: Analysis and quality assessment tools
- `initializers/`: Cube initialization strategies
  - `grid_initializer.py`: Grid-based initialization
  - `ndvi_cluster_initializer.py`: NDVI clustering-based initialization

## Output

Results are automatically saved to:
- `outputs/st_cube_analysis/standard/` - Standard segmentation results
- `outputs/st_cube_analysis/ndvi_clustering/` - NDVI clustering results  
- `outputs/st_cube_analysis/custom/` - Custom segmentation results

Each output directory contains:
- Segmentation maps
- Quality analysis reports
- Statistical summaries
- Visualization files

## Example Scripts

Run the example script to see all methods in action:

```bash
python example_usage.py
```

Or run the main script for a demonstration:

```bash
python segmentation_main.py
```
