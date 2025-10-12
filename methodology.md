---
layout: page
title: Methodology
permalink: /methodology/
---

# Methodology

## Data Sources

**Satellite Imagery**: Landsat 5, 7, and 8 annual composites (1985-2025) acquired through Google Earth Engine. Four spectral bands (Blue, Green, Red, NIR) processed at 30-meter resolution with cloud masking and atmospheric correction applied.

**Study Area**: Barcelona Metropolitan Area encompassing 36 municipalities with diverse urban, suburban, and natural landscapes covering approximately 636 km².

**Vegetation Index**: Normalized Difference Vegetation Index (NDVI) calculated from NIR and Red bands to quantify vegetation health and density across temporal and spatial dimensions.

## Space-Time Cube Framework

The analysis employs a three-dimensional data structure where x and y coordinates represent spatial dimensions and time forms the temporal axis. Each pixel location becomes a vertical trace through time, creating a comprehensive spatiotemporal dataset.

**Trace Construction**: Individual pixel locations tracked across 40+ years, generating temporal profiles of vegetation change. Each trace contains annual NDVI values forming a time series signature.

**Filtering Criteria**: Traces selected based on minimum NDVI thresholds (>0.3) and temporal variance requirements to focus analysis on areas with significant vegetation presence and dynamic behavior.

## Clustering Algorithm

**DBSCAN Implementation**: Density-based spatial clustering applied to combined temporal and spatial feature vectors. The algorithm identifies clusters of similar vegetation behavior without requiring predetermined cluster numbers.

**Feature Engineering**: Multi-dimensional vectors combining normalized NDVI time series with spatial coordinates. Temporal and spatial components weighted according to analysis requirements.

**Parameters**: 
- Epsilon (ε): Defines neighborhood radius for cluster formation
- MinPts: Minimum points required to form dense clusters
- Spatial constraints: Maximum distance thresholds ensure spatial coherence

## Validation Approach

**Spatial Coherence**: Post-processing verification ensures clustered traces maintain geographic proximity within defined distance thresholds.

**Temporal Consistency**: Cluster validation through statistical analysis of NDVI patterns, confirming similar vegetation dynamics within identified groups.

**Municipality-Level Analysis**: Results aggregated and validated at municipal boundaries for administrative relevance and policy application.
