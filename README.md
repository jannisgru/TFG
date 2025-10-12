# Spatiotemporal Analysis of Vegetation Cover in Barcelona

**[View Project Documentation & Results](https://jannisgru.github.io/TFG/)**

A comprehensive analysis framework for monitoring vegetation dynamics using Landsat satellite imagery.

**Academic Context**: Bachelor's thesis project - Universitat Politècnica de Catalunya

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/jannisgru/TFG.git
cd TFG
```

2. **Create conda environment**
```bash
conda env create -f environment.yml
conda activate tfg
```

3. **Install additional dependencies** (if needed)
```bash
pip install -r requirements.txt
```

## Usage

### Data Processing Pipeline

1. **Acquire satellite data** using Google Earth Engine:
```bash
# Copy script content to GEE Code Editor
cat scripts/processing/landsat_data_acquisition.js
# Run in GEE and download to data/raw/
```

2. **Create multidimensional raster**:
```python
python scripts/processing/create_mdim_raster.py
```

3. **Run spatiotemporal analysis**:
```python
python scripts/analysis/segmentation_main.py
```

### Configuration

Edit `config/config.yaml` and `scripts/analysis/segment_config.yaml` to customize:
- Study area boundaries
- NDVI thresholds
- Clustering parameters
- Output formats

### Visualization

Generate interactive visualizations:
```python
# 3D space-time cubes
python scripts/visualization/visualize_interactive.py

# Statistical plots
python scripts/visualization/municipality_ndvi_statistics.py

# JSON cluster viewer
python scripts/visualization/json_cluster_visualizer.py
```

## Quick Start

```bash
# Clone repository
git clone https://github.com/jannisgru/TFG.git
cd TFG

# Setup environment
conda env create -f environment.yml
conda activate tfg

# Run analysis
python scripts/analysis/segmentation_main.py
```

## Project Structure

```
├── scripts/analysis/        # Core segmentation algorithms
├── scripts/processing/      # Data acquisition and preprocessing  
├── scripts/visualization/   # Interactive plotting tools
├── data/boundaries/         # Administrative boundaries
├── config/                  # Configuration files
└── outputs/                 # Results and visualizations
```

## Data Requirements

- **Raw Data**: Landsat annual composites (1984–2025) from Google Earth Engine
- **Bands**: Blue, Green, Red, NIR (30m resolution)
- **Format**: GeoTIFF files named by year (e.g., 1998.tif, 1999.tif, ...)
- **Boundaries**: AMB administrative and municipal boundaries (Shapefile format)

## Key Scripts

- `scripts/processing/landsat_data_acquisition.js`: Google Earth Engine data export
- `scripts/processing/create_mdim_raster.py`: Convert GeoTIFFs to NetCDF time series
- `scripts/analysis/segmentation_main.py`: Main clustering analysis
- `scripts/visualization/visualize_interactive.py`: Generate 3D visualizations

## Configuration Files

- `config/config.yaml`: General project settings
- `scripts/analysis/segment_config.yaml`: Clustering parameters
- `environment.yml`: Conda environment specification