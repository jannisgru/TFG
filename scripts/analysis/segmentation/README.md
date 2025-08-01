# Vegetation-focused ST-Cube Segmentation

An implementation of spatiotemporal cube segmentation for vegetation analysis using NDVI clustering with spatial constraints.

---

## Overview

This package segments satellite NDVI time series into spatially and temporally coherent "cubes" (clusters) representing vegetation patches with similar NDVI dynamics and spatial proximity.

---

## How It Works

1. **Data Loading**: Loads a NetCDF file containing NDVI time series, optionally filtered by municipality.
2. **Vegetation Filtering**: Selects pixels with mean NDVI above a threshold and sufficient temporal variance.
3. **Clustering**: 
   - Combines NDVI time series and spatial coordinates, weighted by `temporal_weight` and `spatial_weight`.
   - Uses DBSCAN or k-means to cluster pixels into NDVI-similar and spatially close groups.
   - Number of clusters is controlled by `n_clusters` (if set), otherwise determined automatically.
4. **Spatial Constraints**: Ensures clusters are spatially coherent and not too dispersed (`max_spatial_distance`).
5. **Cube Creation**: Each cluster is converted into a "cube" with summary statistics and metadata.
6. **Export & Visualization**: Results are exported as a single JSON file per run and visualized using static (Matplotlib) and interactive (Plotly) tools. All outputs for a run are placed in a timestamped subfolder.

---

## Key Parameters

Set in `segment_config.yaml`:

| Parameter                | Description                                                      |
|--------------------------|------------------------------------------------------------------|
| `min_cube_size`          | Minimum pixels for a valid cluster/cube                          |
| `max_spatial_distance`   | Max pixel distance for spatial clustering                        |
| `min_vegetation_ndvi`    | Minimum NDVI to consider as vegetation                           |
| `ndvi_variance_threshold`| Minimum NDVI variance to include a pixel                         |
| `n_clusters`             | Target number of clusters (0/None = automatic)                   |
| `temporal_weight`        | Weight for NDVI vs. spatial features                             |
| `spatial_weight`         | Weight for spatial coordinates in feature space                  |
| `min_samples_ratio`      | Minimum samples as ratio of total pixels for DBSCAN              |
| `eps_search_attempts`    | Attempts to find optimal epsilon for DBSCAN                      |

---

## Output Structure

- **Output Folder**: Each run creates a timestamped subfolder inside your configured output directory.
- **JSON File**: All cluster data for the run is saved as `vegetation_clusters_<municipality>.json`.
- **Visualizations**: Static and interactive visualizations are saved in subfolders within the run's output folder.

---

## Usage

**Basic Example:**
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

---

## Parameter Tuning Tips

- **`temporal_weight` vs `spatial_weight`**: Adjust to prioritize NDVI similarity or spatial proximity.
- **`max_spatial_distance`**: Higher values allow more spatial spread in clusters.
- **`n_clusters`**: Set to 0/None for automatic detection (DBSCAN).
- **`min_cube_size`**: Prevents tiny/noisy clusters.
- **`ndvi_variance_threshold`**: Filters out static vegetation.

---

## Data Requirements

- **NetCDF file** (created using `processing/create_mdim_raster.py`) with:
  - `ndvi` variable: NDVI values (dimensions: time, y, x or time, municipality, y, x)
  - `time`, `y`, `x` dimensions
  - Optional `municipality` dimension

---

## License

This project is part of a university thesis focusing on vegetation analysis using satellite time series data.

---