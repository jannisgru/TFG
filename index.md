---
layout: page
title: Home
permalink: /
---

# Spatiotemporal Analysis of Vegetation Cover in Barcelona

This study analyzes the evolution of vegetation cover across the Barcelona Metropolitan Area from 1984 to 2025 using satellite imagery. Annual maximum NDVI (Normalized Difference Vegetation Index) composites were generated to represent peak greenness each year, and these layers were assembled into a three-dimensional space–time cube that preserves both spatial and temporal continuity.

Remote sensing enables efficient, large-scale monitoring of vegetation: indices like NDVI (derived from red and near-infrared reflectances) provide a convenient measure of plant "greenness" and health. The aggregated results reveal clear trends: the analysis identified widespread vegetation decline (browning) from the 1980s through the early 2000s, followed by renewed urban greening after the 2010s.

## Key Findings

- **1980s–2000s**: Widespread vegetation decline during rapid urban expansion
- **Post-2010s**: Renewed greening through municipal initiatives and reduced construction
- **Protected Areas**: Stable vegetation cover in natural parks and agricultural zones
- **Infrastructure Impact**: Clear NDVI declines associated with major construction projects

## Technical Approach

- **Data Source**: Landsat 5, 7, and 8 satellite imagery (1984–2025)
- **Processing**: Google Earth Engine for cloud masking and atmospheric correction
- **Analysis**: Space-time cube framework with DBSCAN clustering
- **Output**: Interactive visualizations and spatiotemporal trend analysis

## Quick Start

```bash
git clone https://github.com/jannisgru/TFG.git
cd TFG
conda env create -f environment.yml
conda activate tfg
python scripts/analysis/segmentation_main.py
```

---

*Bachelor's thesis project - Universitat Politècnica de Catalunya*