# Barcelona Metropolitan Area (AMB) Vegetation Analysis

Spatiotemporal analysis of vegetation dynamics in the Barcelona Metropolitan Area using Landsat data from 1985-2025.

## Project Structure

```
TFG/
├── data/
│   ├── raw/                 # Landsat annual composites
│   ├── processed/           # Processed NDVI data
│   └── boundaries/          # AMB and municipal boundaries
├── config/                  # Configuration files
└── outputs/                 # Analysis results and maps
```

## Setup

Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate amb-vegetation-analysis
```

## Data Requirements

- **Raw Data**: Landsat annual composites (1998-2025) from Google Earth Engine
- **Bands**: Blue, Green, Red, NIR (30m resolution)
- **Format**: GeoTIFF files named by year (1998.tif, 1999.tif, etc.)
- **Projection**: EPSG:4326 (WGS84) - will be reprojected to EPSG:25831
- **Boundaries**: AMB administrative boundaries and municipal boundaries

## Data Acquisition

The Google Earth Engine script for data acquisition is located in `scripts/landsat_data_acquisition.js`:

1. Copy the script to GEE Code Editor
2. Run to export annual composites to Google Drive  
3. Download files and place in `data/raw/`
4. Files should be named: 1998.tif, 1999.tif, ..., 2025.tif

## Basic Usage

1. Place Landsat data in `data/raw/`
2. Add boundary files to `data/boundaries/`
3. Configure parameters in `config/config.yaml`
4. Run analysis scripts or notebooks

---

*Bachelors's thesis project on vegetation change detection using space-time cube methodology.*
