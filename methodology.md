---
layout: page
title: Methodology
permalink: /methodology/
---

# Methodology

The analysis integrates multi-decadal Landsat satellite data with advanced spatiotemporal processing. Surface reflectance imagery from Landsat 5, 7, and 8 was accessed via Google Earth Engine to provide continuous coverage over 1984–2025.

## Data Processing

**Annual Composites**: For each year, the maximum NDVI value at each pixel was selected to form a "peak greenness" mosaic, which minimizes seasonal and cloud effects. These yearly NDVI composites were stacked into a Space–Time Cube (STC) so that each geolocated "trace" contains the full NDVI time series.

**Quality Control**: Cloud masking and atmospheric correction applied to all imagery. Surface reflectance products ensure consistent radiometric calibration across the multi-sensor time series.

**Study Area**: Barcelona Metropolitan Area encompassing 36 municipalities with diverse urban, suburban, and natural landscapes covering approximately 636 km².

## Space-Time Cube Framework

After assembling the STC, data cleaning removed non-vegetated or unchanging locations. Each remaining pixel's NDVI trace was then encoded as a feature vector combining its temporal NDVI values and spatial coordinates.

**Trace Construction**: Individual pixel locations tracked across 40+ years, generating temporal profiles of vegetation change. Each trace contains annual maximum NDVI values forming a time series signature.

**Filtering Criteria**: Traces selected based on minimum NDVI thresholds and temporal variance requirements to focus analysis on areas with significant vegetation presence and dynamic behavior.

## Spatiotemporal Clustering

**DBSCAN Implementation**: Density-based clustering applied to group pixels with similar NDVI trends into spatially coherent clusters. DBSCAN was chosen because it can discover irregularly shaped clusters without predefining a number of clusters.

**Feature Engineering**: Multi-dimensional vectors combining normalized NDVI time series with spatial coordinates. Temporal and spatial components weighted according to analysis requirements.

**Cluster Validation**: The resulting clusters were characterized by metrics such as spatial extent and average NDVI profile. Spatial coherence verification ensures clustered traces maintain geographic proximity within defined distance thresholds.

## Output Generation

Final products (cluster maps and trend graphs) were exported for visualization and further analysis. Interactive 3D visualizations using Plotly enable exploration of spatiotemporal patterns, while statistical dashboards provide municipality-level insights.
