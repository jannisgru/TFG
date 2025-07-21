# Vegetation-focused ST-Cube Segmentation

A streamlined implementation of spatiotemporal cube segmentation focused on vegetation analysis using NDVI clustering with local spatial constraints.

## Key Features

- **Vegetation-focused NDVI Clustering**: Groups nearby pixels with similar NDVI time series, focusing only on vegetation areas (NDVI ≥ 0.4).
- **Local Spatial Constraints**: Enforces a maximum spatial distance (e.g., 10 pixels) for clustering to capture local vegetation patterns.
- **Connected Component Analysis**: Ensures spatial connectivity within clusters.
- **Summary Visualizations**: Generates summary plots (histograms, scatter plots, pie charts) for cluster statistics using Matplotlib and Seaborn.
- **Performance Optimizations**: Uses chunked data loading and memory-efficient processing with Dask and Xarray.
- **Parameterizable Segmentation**: All main parameters (NDVI threshold, spatial distance, cluster size, number of clusters, etc.) are configurable.

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

# With custom parameters
params = VegetationSegmentationParameters(
    max_spatial_distance=10,      # Local clustering within 10 pixels
    min_vegetation_ndvi=0.4,      # Focus on areas with NDVI ≥ 0.4
    min_cube_size=20,             # Minimum 20 pixels per cluster
    ndvi_variance_threshold=0.01, # Filter out static areas
    n_clusters=5,                 # Target number of clusters
    temporal_weight=0.7           # Weight for temporal vs spatial similarity
)

vegetation_cubes = segment_vegetation(
    netcdf_path="data/landsat_multidimensional_Sant_Marti.nc",
    parameters=params,
    municipality_name="Sant Martí",
    create_visualizations=True,
    output_dir="outputs/vegetation_clustering"
)
```

## Parameters

### VegetationSegmentationParameters

- `max_spatial_distance` (default: 10): Maximum distance in pixels for spatial connectivity.
- `min_vegetation_ndvi` (default: 0.4): Minimum NDVI value to consider as vegetation.
- `min_cube_size` (default: 20): Minimum number of pixels required for a valid cluster.
- `ndvi_variance_threshold` (default: 0.01): Minimum NDVI variance over time to consider a pixel as dynamic vegetation.
- `n_clusters` (default: 10): Target number of clusters for k-means clustering.
- `temporal_weight` (default: 0.7): Weight for temporal (NDVI) vs spatial similarity in clustering.

## How It Works

1. **Data Loading**: Loads NetCDF file using Xarray with chunking for memory efficiency.
2. **Vegetation Filtering**: Identifies pixels with NDVI ≥ threshold and sufficient temporal variance.
3. **Clustering**: Combines standardized NDVI time series and normalized spatial coordinates, weighted by `temporal_weight`, and clusters using k-means.
4. **Spatial Constraints**: Filters clusters to ensure spatial coherence and minimum size.
5. **Cube Creation**: Computes statistics (mean NDVI, seasonality, trend, type) for each cluster.
6. **Visualization**: Optionally generates summary plots for cluster size, NDVI, seasonality, and vegetation type.

## Output

- **List of cluster dictionaries**: Each with spatial coordinates, NDVI profiles, area, mean NDVI, seasonality, trend, and vegetation type.
- **Summary Plots** (if enabled): Saved as PNG in the output directory, including:
  - Cluster size distribution
  - Mean NDVI distribution
  - Seasonality vs NDVI scatter
  - Vegetation type distribution

## Data Format

The package expects NetCDF files with:
- `ndvi` variable: NDVI values with dimensions (time, y, x) or (time, municipality, y, x)
- `time` dimension: Temporal axis
- `y`, `x` dimensions: Spatial coordinates
- Optional `municipality` dimension for multi-municipality datasets

## Example Output

```
=== Starting Vegetation NDVI Clustering Segmentation ===
Data: data/landsat_multidimensional_Sant_Marti.nc
Municipality: Sant Martí
1. Loading and validating data...
Loaded dataset with shape: {'time': 139, 'y': 50, 'x': 80}
Valid pixels: 2847/4000 (71.2%)
2. Extracting vegetation pixels...
Found 1523 vegetation pixels
Mean NDVI range: 0.445 - 0.731
3. Performing spatially-constrained clustering...
Clustering 1523 vegetation pixels...
Created 8 spatially-constrained clusters
4. Creating vegetation ST-cubes...
5. Creating visualizations...
Summary plots saved to: outputs/vegetation_clustering/vegetation_summary_Sant Martí.png
=== Segmentation completed: 8 clusters ===
```

## Files

- `segmentation_main.py`: Main script with `segment_vegetation()` function and core logic.
- `base.py`: VegetationSegmentationParameters configuration class.
- `cube.py`: STCube data structure for spatiotemporal segments.
- `interactive_visualization.py`: (Optional) Interactive HTML visualizations using Plotly.
- `initializers/ndvi_cluster_initializer.py`: (Optional) NDVI clustering-based initialization.
- `test_vegetation_clustering.py`: Test script for vegetation clustering.
- `README.md`: This documentation.

## License

This project is part of a university thesis focusing on vegetation analysis using satellite time series data.

## Example Scripts

Run the main script for a demonstration:

```bash
python segmentation_main.py
```
