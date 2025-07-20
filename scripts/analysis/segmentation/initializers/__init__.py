"""
Vegetation-focused initialization for ST-Cube segmentation.

This package contains vegetation-focused initialization strategies for creating
spatiotemporal cubes from satellite data.
"""

from .ndvi_cluster_initializer import VegetationNDVIClusteringInitializer

__all__ = ['VegetationNDVIClusteringInitializer']
