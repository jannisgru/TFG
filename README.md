# Spatiotemporal Vegetation Analysis of Barcelona Metropolitan Area

A comprehensive analysis framework for monitoring vegetation dynamics in the Barcelona Metropolitan Area using four decades of Landsat satellite data (1985-2025).

## Overview

This research presents a unique approach to understanding urban vegetation patterns through spatiotemporal analysis of satellite time series data. The methodology combines space-time cube analysis with advanced clustering techniques to identify and track vegetation communities across temporal and spatial dimensions.

**Key Features:**
- Automated processing of 40+ years of Landsat imagery
- Space-time cube segmentation using DBSCAN clustering
- Interactive 3D visualizations of vegetation dynamics
- Municipal-level vegetation trend analysis
- Comprehensive NDVI time series analysis

## Methodology

The analysis pipeline implements a three-stage approach:

1. **Data Processing**: Automated acquisition and normalization of Landsat annual composites
2. **Spatiotemporal Segmentation**: DBSCAN-based clustering of NDVI time series with spatial constraints
3. **Visualization**: Interactive 3D space-time cubes and statistical dashboards

## Results & Visualizations

**Interactive Examples:**
- [3D Spatiotemporal Visualization (Sant Martí)](https://jannisgru.github.io/TFG/outputs/3d_spatiotemporal_Sant_Mart%C3%AD.html)
- [Additional Examples](https://jannisgru.github.io/TFG/examples/) (Coming Soon)

**Analysis Outputs:**
- [Methodology Documentation](https://jannisgru.github.io/TFG/scripts/analysis/)
- [Technical Implementation](https://jannisgru.github.io/TFG/technical/) (Coming Soon)

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

## Academic Context

This work was developed as part of a Bachelor's thesis at Universitat Politècnica de Catalunya, focusing on the application of space-time cube methodology for environmental monitoring in metropolitan areas.

**Keywords:** Remote Sensing, NDVI, Space-Time Analysis, Urban Vegetation, DBSCAN Clustering
