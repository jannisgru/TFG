# Vegetation-focused ST-Cube Segmentation

Spatiotemporal cube segmentation for vegetation analysis using NDVI clustering with spatial constraints.

## Overview

This package segments satellite NDVI time series into spatially and temporally coherent clusters representing vegetation patches with similar NDVI dynamics and spatial proximity.

## How It Works

1. **Data Loading**: Loads NetCDF files containing NDVI time series, with optional municipality filtering
2. **Vegetation Filtering**: Selects pixels with mean NDVI above threshold and sufficient temporal variance
3. **Clustering**: 
   - Combines NDVI time series and spatial coordinates, weighted by `temporal_weight` and `spatial_weight`
   - Uses DBSCAN to cluster pixels into NDVI-similar and spatially close groups
   - Number of clusters controlled by `n_clusters` (if set), otherwise determined automatically
4. **Spatial Bridging**: 
   - Optional post-processing step that merges clusters connected by "bridges" of similar pixels
   - Allows clusters to grow through indirect connections, overcoming strict spatial distance limits
   - Only merges clusters—never splits them
5. **Spatial Constraints**: Ensures clusters remain spatially coherent within `max_spatial_distance`
6. **Export & Visualization**: Results exported as JSON with static (Matplotlib) and interactive (Plotly) visualizations

## Clustering Algorithm (DBSCAN)

The core segmentation uses DBSCAN clustering on combined NDVI and spatial features:

1. **Feature Construction**: Each pixel gets a feature vector combining NDVI time series and spatial coordinates (x, y)
2. **DBSCAN Clustering**: 
   - Density-based clustering that groups pixels close in feature space
   - Finds arbitrarily shaped clusters automatically
   - Marks outliers/noise pixels
3. **Parameters**: Controlled by `min_samples_ratio` (minimum cluster size) and `eps_search_attempts` (neighborhood radius optimization)

DBSCAN is ideal for flexible, shape-adaptive clustering without requiring pre-specified cluster counts.

## Spatial Bridging Algorithm

Post-processing step using graph-based approach with NetworkX:

1. **Graph Construction**: All pixels represented as nodes with edges to neighboring pixels (within `connectivity_radius`)
2. **Bridge Search**: For each cluster pair, searches for valid connecting paths using graph traversal
3. **Bridge Validation**: Path must satisfy:
   - NDVI difference between consecutive pixels ≤ `bridge_similarity_tolerance`
   - ≤ `max_bridge_gap` consecutive dissimilar pixels allowed
   - Overall similar pixel density ≥ `min_bridge_density`
   - Total length ≤ `max_bridge_length`
   - Only considers clusters ≥ `min_cluster_size_for_bridging`
4. **Merging**: Valid bridges merge connected clusters

Enable for fragmented landscapes; disable for compact, strictly spatial clusters.

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

### Spatial Bridging Parameters
| Parameter | Description |
|-----------|-------------|
| `enable_spatial_bridging` | Enable/disable bridging post-processing |
| `bridge_similarity_tolerance` | Max NDVI difference for bridge pixels (0-1) |
| `max_bridge_gap` | Max consecutive dissimilar pixels in bridge |
| `min_bridge_density` | Minimum % similar pixels in bridge (0-1) |
| `connectivity_radius` | Neighborhood size for spatial graph |
| `max_bridge_length` | Maximum bridge path length |
| `min_cluster_size_for_bridging` | Minimum cluster size for bridging eligibility |

## Usage

**Basic:**
```python
from segmentation_main import segment_vegetation
cubes = segment_vegetation()
```

**Custom Parameters:**
```python
from segmentation_main import segment_vegetation
from base import VegetationSegmentationParameters

params = VegetationSegmentationParameters(
    max_spatial_distance=8,
    min_vegetation_ndvi=0.45,
    min_cube_size=20,
    ndvi_variance_threshold=0.02,
    n_clusters=10,
    temporal_weight=0.5
)

cubes = segment_vegetation(
    netcdf_path="data/landsat_multidimensional_Sant_Marti.nc",
    parameters=params,
    municipality_name="Sant Martí",
    create_visualizations=True,
    output_dir="outputs/landsat_multidimensional_Sant_Marti"
)
```

**Command Line:**
```bash
python segmentation_main.py
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