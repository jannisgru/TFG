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
   - Uses K-means to cluster pixels into NDVI-similar and spatially close groups
   - Number of clusters controlled by `n_clusters` parameter
4. **Spatial Constraints**: Ensures clusters remain spatially coherent within `max_spatial_distance`
5. **Export & Visualization**: Results exported as JSON with static (Matplotlib) and interactive (Plotly) visualizations

## Clustering Algorithm (K-means)

The core segmentation uses K-means clustering on combined NDVI and spatial features:

1. **Feature Construction**: Each pixel gets a feature vector combining NDVI time series and spatial coordinates (x, y)
2. **K-means Clustering**: 
   - Partitional clustering that groups pixels into k clusters based on feature similarity
   - Minimizes within-cluster sum of squares
   - Requires pre-specified number of clusters (`n_clusters`)
3. **Parameters**: Number of clusters controlled by `n_clusters`, with spatial coherence enforced via `max_spatial_distance`

K-means is efficient for large datasets and produces compact, spherical clusters with consistent sizes.

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
| `n_clusters` | Target cluster count (required for K-means) |

### Clustering Parameters
| Parameter | Description |
|-----------|-------------|
| `spatial_weight` | Weight for spatial coordinates in feature space |
| `temporal_weight` | Weight for temporal vs spatial features (0-1) |
| `random_state` | Random seed for K-means reproducibility |
| `n_init` | Number of K-means initializations |


## File Structure

```
analysis/
├── __init__.py
├── visualization/
│   ├── __init__.py
│   ├── visualization_2d.py    # Matplotlib static plots
│   ├── visualization_3d.py    # Plotly interactive 3D plots
│   └── common.py              # Common visualization utilities
├── config_loader.py           # YAML config loader
├── json_exporter.py           # JSON export utilities
├── segmentation_main.py       # Main pipeline entry point (contains VegetationSegmentationParameters)
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
- **`n_clusters`**: Must be specified for K-means; determines final number of clusters
- **`min_cube_size`**: Prevents tiny/noisy clusters through post-processing
- **`ndvi_variance_threshold`**: Filters static vegetation

## Data Requirements

NetCDF file (created using `processing/create_mdim_raster.py`) with:
- `ndvi` variable: NDVI values (dimensions: time, y, x or time, municipality, y, x)
- `time`, `y`, `x` dimensions
- Optional `municipality` dimension

## License

University thesis project for vegetation analysis using satellite time series data.
