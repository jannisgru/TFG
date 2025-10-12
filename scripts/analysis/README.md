# Space-Time Cube Segmentation Module

Advanced spatiotemporal clustering algorithm for vegetation time series analysis using NDVI data with spatial and temporal constraints.

## Algorithm Overview

This module implements a density-based clustering approach that identifies coherent vegetation communities by analyzing both temporal NDVI patterns and spatial proximity. The algorithm automatically determines optimal cluster numbers while maintaining spatial coherence across the study area.

## Live Example

[Interactive 3D Visualization - Sant Martí District](https://jannisgru.github.io/TFG/outputs/3d_spatiotemporal_Sant_Mart%C3%AD.html)

## Implementation

The segmentation pipeline consists of five core stages:

### 1. Data Preprocessing
- NetCDF time series loading with municipality filtering
- Vegetation trace selection based on NDVI thresholds and temporal variance

### 2. Feature Engineering
- Multi-dimensional feature vectors combining temporal NDVI signatures and spatial coordinates
- Configurable weighting between temporal and spatial components

### 3. DBSCAN Clustering
- Density-based clustering for automatic cluster number determination
- Noise detection and outlier filtering
- Parameter optimization through `eps` (neighborhood radius) and `min_pts` (minimum cluster size)

### 4. Spatial Validation
- Post-processing spatial coherence verification
- Distance-based cluster refinement

### 5. Output Generation
- JSON export with cluster metadata
- Interactive 3D visualizations using Plotly
- Static analysis plots using Matplotlib

## Configuration

All parameters are configured through `segment_config.yaml`:

**Core Parameters:**
- `eps`: DBSCAN neighborhood radius for cluster formation
- `min_pts`: Minimum points required to form a dense cluster
- `temporal_weight` / `spatial_weight`: Feature weighting balance
- `min_vegetation_ndvi`: NDVI threshold for vegetation identification
- `max_spatial_distance`: Spatial coherence constraint


## Usage

```python
from segmentation_main import run_segmentation

# Run analysis on specific municipality
results = run_segmentation(
    netcdf_path="data/processed/ndvi_timeseries.nc",
    municipality="Sant Martí",
    config_path="segment_config.yaml"
)
```

## Module Structure

```
analysis/
├── segmentation_main.py      # Main entry point
├── segmentation_engine.py    # Core clustering algorithms
├── config_loader.py          # Configuration management
├── json_exporter.py          # Results export
├── visualization/            # Plotting modules
└── segment_config.yaml       # Parameter configuration
```

## Input Requirements

**NetCDF Format:**
- `ndvi` variable with dimensions: (time, y, x) or (time, municipality, y, x)
- Coordinate variables: `time`, `y`, `x`
- Optional: `municipality` dimension for regional filtering

**Generated using:** `scripts/processing/create_mdim_raster.py`