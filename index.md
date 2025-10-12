---
layout: page
title: TFG
permalink: /
---

# Spatiotemporal Analysis of Vegetation Cover in Barcelona

A comprehensive analysis framework for monitoring vegetation dynamics in the Barcelona Metropolitan Area using four decades of Landsat satellite data (1985-2025).

## Overview

This research presents a unique approach to understanding urban vegetation patterns through spatiotemporal analysis of satellite time series data. The methodology combines space-time cube analysis with advanced clustering techniques to identify and track vegetation communities across temporal and spatial dimensions.

**Features:**
- Automated processing of 40+ years of Landsat imagery
- Space-time cube segmentation using DBSCAN clustering
- Interactive 3D visualizations of vegetation dynamics
- Municipal-level vegetation trend analysis
- Comprehensive NDVI time series analysis

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

## Academic Context

This work was developed as part of a Bachelor's thesis at Universitat Politècnica de Catalunya, focusing on the application of space-time cube methodology for environmental monitoring in metropolitan areas.