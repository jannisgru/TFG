# Space-Time Cube Segmentation Module

**[View Full Documentation & Examples](https://jannisgru.github.io/TFG/methodology/)**

Advanced spatiotemporal clustering algorithm for vegetation time series analysis using NDVI data with spatial and temporal constraints.

## Quick Start

```python
from segmentation_main import run_segmentation

# Basic usage
results = run_segmentation(
    netcdf_path="data/processed/ndvi_timeseries.nc",
    municipality="Sant Martí"
)

# With custom configuration
results = run_segmentation(
    netcdf_path="data/processed/ndvi_timeseries.nc",
    municipality="Barcelona",
    config_path="custom_segment_config.yaml",
    output_dir="custom_outputs/"
)
```

## Core Functions

### `run_segmentation()`
Main entry point for spatiotemporal analysis.

**Parameters:**
- `netcdf_path` (str): Path to NetCDF file with NDVI time series
- `municipality` (str, optional): Municipality name for filtering
- `config_path` (str, optional): Custom configuration file path
- `output_dir` (str, optional): Output directory for results

**Returns:** Dictionary with cluster results and file paths

### `VegetationSegmentationEngine`
Core clustering algorithm implementation.

```python
from segmentation_engine import VegetationSegmentationEngine

engine = VegetationSegmentationEngine(config)
clusters = engine.segment_vegetation_traces(ndvi_data, coords)
```

### Visualization Functions

```python
from visualization.visualization_3d import create_interactive_3d_plot
from visualization.visualization_2d import create_cluster_maps

# Generate 3D interactive plot
create_interactive_3d_plot(clusters, output_path="3d_plot.html")

# Generate static maps
create_cluster_maps(clusters, output_dir="maps/")
```

## Configuration

Edit `segment_config.yaml` to customize analysis parameters:

```yaml
# Clustering parameters
clustering:
  eps: 0.5                    # DBSCAN neighborhood radius
  min_pts: 5                  # Minimum points per cluster
  temporal_weight: 0.7        # Weight for NDVI time series
  spatial_weight: 0.3         # Weight for spatial coordinates

# Filtering parameters
filtering:
  min_vegetation_ndvi: 0.3    # Minimum NDVI threshold
  ndvi_variance_threshold: 0.02  # Minimum temporal variance
  max_spatial_distance: 1000  # Maximum cluster spatial extent (m)

# Output settings
output:
  enable_3d_visualization: true
  enable_static_plots: true
  export_json: true
```

## Command Line Interface

```bash
# Run with default settings
python segmentation_main.py

# Specify municipality and config
python segmentation_main.py --municipality "Barcelona" --config custom_config.yaml

# Custom output directory
python segmentation_main.py --output-dir results/experiment_1/
```

## Module Structure

```
analysis/
├── segmentation_main.py      # Main entry point and CLI
├── segmentation_engine.py    # Core clustering algorithms  
├── config_loader.py          # YAML configuration loader
├── json_exporter.py          # Results export utilities
├── visualization/
│   ├── visualization_2d.py   # Static matplotlib plots
│   ├── visualization_3d.py   # Interactive plotly visualizations
│   └── common.py            # Shared plotting utilities
└── segment_config.yaml       # Default configuration
```

## Data Format Requirements

**Input NetCDF:**
```python
# Required variables and dimensions
ndvi: (time, y, x) or (time, municipality, y, x)
time: datetime64 array
y, x: coordinate arrays (UTM or geographic)

# Optional variables
municipality: string labels for administrative boundaries
```

**Generated using:**
```bash
python scripts/processing/create_mdim_raster.py
```

## Dependencies

- **Core**: numpy, pandas, xarray, scikit-learn
- **Visualization**: matplotlib, plotly, geopandas
- **I/O**: netCDF4, pyyaml, json

Install via: `conda env create -f environment.yml`