# Vegetation-focused ST-Cube Segmentation

Spatiotemporal cube segmentation for vegetation analysis using NDVI clustering with spatial constraints.

## Overview

This package segments satellite NDVI time series into spatially and temporally coherent clusters representing vegetation patches with similar NDVI dynamics and spatial proximity. Uses DBSCAN clustering for robust density-based segmentation that automatically determines the number of clusters.

## Example Interactive Visualization

- [View 3D Spatiotemporal Example (Sant Martí)](https://jannisgru.github.io/TFG/outputs/3d_spatiotemporal_Sant_Mart%C3%AD.html)

## How It Works

1. **Data Loading**: Loads NetCDF files containing NDVI time series, with optional municipality filtering
2. **Vegetation Filtering**: Selects pixels with mean NDVI above threshold and sufficient temporal variance
3. **Clustering**: 
   - Combines NDVI time series and spatial coordinates, weighted by `temporal_weight` and `spatial_weight`
   - Uses DBSCAN to cluster pixels into NDVI-similar and spatially close groups
   - Number of clusters determined automatically by DBSCAN parameters (`eps` and `min_samples`)
4. **Spatial Constraints**: Ensures clusters remain spatially coherent within `max_spatial_distance`
5. **Export & Visualization**: Results exported as JSON with static (Matplotlib) and interactive (Plotly) visualizations

## Clustering Algorithm (DBSCAN)

The core segmentation uses DBSCAN clustering on combined NDVI and spatial features:

1. **Feature Construction**: Each pixel gets a feature vector combining NDVI time series and spatial coordinates (x, y)
2. **DBSCAN Clustering**: 
   - Density-based clustering that groups pixels based on feature similarity and density
   - Automatically determines the number of clusters based on data density
   - Can identify and filter out noise points (outliers)
   - Controlled by `eps` (maximum distance between neighbors) and `min_samples` (minimum points to form cluster)
3. **Parameters**: Cluster formation controlled by `eps` and `min_samples`, with spatial coherence enforced via `max_spatial_distance`

DBSCAN is robust to noise and can find clusters of varying shapes and sizes, making it well-suited for irregular vegetation patterns.

## Key Parameters

Configure in `segment_config.yaml`:

### Segmentation Parameters
| Parameter | Description |
|-----------|-------------|
| `min_cube_size` | Minimum pixels for valid cluster |
| `max_spatial_distance` | Maximum pixel distance for spatial clustering |
| `min_vegetation_ndvi` | Minimum NDVI threshold for vegetation |
| `ndvi_variance_threshold` | Minimum NDVI variance to include pixel |

### Clustering Parameters
| Parameter | Description |
|-----------|-------------|
| `eps` | DBSCAN eps: maximum distance between samples to be neighbors |
| `min_samples` | DBSCAN min_samples: minimum samples in neighborhood to form cluster |
| `temporal_weight` | Weight for temporal vs spatial features (0-1) |
| `spatial_weight` | Weight for spatial coordinates in feature space |


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

**Segmentation Parameters:**
- **`max_spatial_distance`**: Higher values allow more spatial spread in post-processing
- **`min_cube_size`**: Prevents tiny/noisy clusters through post-processing  
- **`min_vegetation_ndvi`**: Threshold for initial vegetation pixel selection
- **`ndvi_variance_threshold`**: Filters static vegetation pixels

**Clustering Parameters:**
- **`eps`**: Controls cluster density - smaller values create tighter, more clusters
- **`min_samples`**: Minimum points needed to form a cluster - higher values reduce noise but may merge small clusters
- **`temporal_weight`**: Weight for NDVI time series features (higher = more NDVI similarity focus)
- **`spatial_weight`**: Weight for spatial coordinates (higher = more spatial compactness)

### DBSCAN Parameter Guidelines
- Start with `eps=1.0`, `min_samples=20`, `temporal_weight=0.1`, `spatial_weight=0.4` for initial testing
- If too many small clusters: increase `eps` or decrease `min_samples`
- If too few large clusters: decrease `eps` or increase `min_samples`  
- Balance temporal vs spatial features: increase `temporal_weight` for more NDVI similarity, increase `spatial_weight` for more compact clusters
- Monitor noise points in logs - too many may indicate poor parameter tuning

## Data Requirements

NetCDF file (created using `processing/create_mdim_raster.py`) with:
- `ndvi` variable: NDVI values (dimensions: time, y, x or time, municipality, y, x)
- `time`, `y`, `x` dimensions
- Optional `municipality` dimension

## License

University thesis project for vegetation analysis using satellite time series data.
