"""
Vegetation-focused ST-Cube Segmentation Package

A streamlined implementation of spatiotemporal cube segmentation focused specifically
on vegetation analysis using NDVI clustering with local spatial constraints.

Main Components:
- VegetationNDVIClusteringInitializer: Creates spatially-aware NDVI clusters for vegetation
- VegetationSegmentationParameters: Simplified parameter structure
- STCube: Individual spatiotemporal cube representation
- InteractiveVisualization: HTML-based Plotly visualizations

Main Function:
- segment_vegetation: Run vegetation-focused NDVI clustering segmentation

Example Usage:
    # Vegetation NDVI clustering segmentation
    from segmentation import segment_vegetation, VegetationSegmentationParameters
    
    params = VegetationSegmentationParameters(
        max_spatial_distance=10,
        min_vegetation_ndvi=0.4,
        min_cube_size=20
    )
    
    cubes = segment_vegetation("data.nc", params)
"""

from .base import VegetationSegmentationParameters
from .cube import STCube
from .initializers import VegetationNDVIClusteringInitializer
from .interactive_visualization import InteractiveVisualization

# Import main function for easy access
from .segmentation_main import segment_vegetation

__all__ = [
    # Main API
    'segment_vegetation',
    
    # Core classes
    'VegetationSegmentationParameters',
    'STCube',
    'VegetationNDVIClusteringInitializer',
    'InteractiveVisualization',
]

__version__ = '3.0.0-vegetation'
__author__ = 'Vegetation Segmentation Team'
