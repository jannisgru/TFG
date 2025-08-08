# Vegetation-focused ST-Cube Segmentation

Spatiotemporal cube segmentation for vegetation analysis using NDVI clustering with spatial constraints.

## Overview

This package segments satellite NDVI time series into spatially and temporally coherent clusters representing vegetation patches with similar NDVI dynamics and spatial proximity.

## Example Interactive Visualization

- [View 3D Spatiotemporal Example (Sant Martí)](https://jannisgru.github.io/TFG/outputs/3d_spatiotemporal_Sant_Mart%C3%AD.html)

## How It Works

1. **Data Loading**: Loads NetCDF files containing NDVI time series, with optional municipality filtering
2. **Vegetation Filtering**: Selects pixels with mean NDVI above threshold and sufficient temporal variance
3. **Clustering**: 
   - Combines NDVI time series and spatial coordinates, weighted by `temporal_weight` and `spatial_weight`
   - Uses DBSCAN to cluster pixels into NDVI-similar and spatially close groups
   - Number of clusters controlled by `n_clusters` (if set), otherwise determined automatically
4. **Spatial Constraints**: Ensures clusters remain spatially coherent within `max_spatial_distance`
5. **Export & Visualization**: Results exported as JSON with static (Matplotlib) and interactive (Plotly) visualizations

## Clustering Algorithm (DBSCAN)

The core segmentation uses DBSCAN clustering on combined NDVI and spatial features:

1. **Feature Construction**: Each pixel gets a feature vector combining NDVI time series and spatial coordinates (x, y)
2. **DBSCAN Clustering**: 
   - Density-based clustering that groups pixels close in feature space
   - Finds arbitrarily shaped clusters automatically
   - Marks outliers/noise pixels
3. **Parameters**: Controlled by `min_samples_ratio` (minimum cluster size) and `eps_search_attempts` (neighborhood radius optimization)

DBSCAN is ideal for flexible, shape-adaptive clustering without requiring pre-specified cluster counts.

## Key Parameters

Configure in `segment_config.yaml`:

### Segmentation Parameters
| Parameter | Description |
|-----------|-------------|
| `min_cube_size` | Minimum pixels for valid cluster |
| `max_spatial_distance` | Maximum pixel distance for spatial clustering |
| `min_vegetation_ndvi` | Minimum NDVI threshold for vegetation |
| `ndvi_variance_threshold` | Minimum NDVI variance to include pixel |
| `temporal_weight` | Weight for NDVI vs. spatial features |
| `n_clusters` | Target cluster count (0/None = automatic) |

### Clustering Parameters
| Parameter | Description |
|-----------|-------------|
| `spatial_weight` | Weight for spatial coordinates in feature space |
| `min_samples_ratio` | Minimum samples ratio for DBSCAN |
| `eps_search_attempts` | Attempts to find optimal DBSCAN epsilon |


## File Structure

```
segmentation/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── base.py                # Parameter dataclass
│   └── cube.py                # STCube and CubeCollection
├── visualization/
│   ├── __init__.py
│   ├── static.py              # Matplotlib static plots
│   └── interactive.py         # Plotly interactive 3D plots
├── initializers/
│   ├── __init__.py
│   └── ndvi_cluster_initializer.py  # NDVI clustering logic
├── config_loader.py           # YAML config loader
├── json_exporter.py           # JSON export utilities
├── segmentation_main.py       # Main pipeline entry point
├── segment_config.yaml        # Main YAML config
└── README.md
```

## Output Structure

- **Output Folder**: Timestamped subfolder per run
- **JSON File**: All cluster data saved as `vegetation_clusters_<municipality>.json`
- **Visualizations**: Static and interactive plots in subfolders

## Parameter Tuning Tips

- **`temporal_weight` vs `spatial_weight`**: Balance NDVI similarity vs. spatial proximity
- **`max_spatial_distance`**: Higher values allow more spatial spread
- **`n_clusters`**: Set to 0/None for automatic detection
- **`min_cube_size`**: Prevents tiny/noisy clusters
- **`ndvi_variance_threshold`**: Filters static vegetation

## Data Requirements

NetCDF file (created using `processing/create_mdim_raster.py`) with:
- `ndvi` variable: NDVI values (dimensions: time, y, x or time, municipality, y, x)
- `time`, `y`, `x` dimensions
- Optional `municipality` dimension

## License

University thesis project for vegetation analysis using satellite time series data.
