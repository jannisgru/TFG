"""
Vegetation-focused ST-Cube Segmentation Package

A streamlined implementation of spatiotemporal cube segmentation focused specifically
on vegetation analysis using NDVI clustering with local spatial constraints.

Main Components:
- VegetationNDVIClusteringInitializer: Creates spatially-aware NDVI clusters for vegetation
- VegetationSegmentationParameters: Simplified parameter structure
- STCube: Individual spatiotemporal cube representation
- InteractiveVisualization: HTML-based Plotly visualizations
- StaticVisualization: Publication-ready matplotlib visualizations
- ConfigLoader: Centralized configuration management

Main Function:
- segment_vegetation: Run vegetation-focused NDVI clustering segmentation

Example Usage:
    # Vegetation NDVI clustering segmentation with config
    from segmentation import segment_vegetation, VegetationSegmentationParameters, get_config
    
    # Use default config values
    cubes = segment_vegetation()
    
    # Or customize specific parameters
    params = VegetationSegmentationParameters(
        max_spatial_distance=10,
        min_vegetation_ndvi=0.4,
        min_cube_size=20
    )
    
    cubes = segment_vegetation(parameters=params)
"""
from .base import VegetationSegmentationParameters
from .cube import STCube
from .initializers import VegetationNDVIClusteringInitializer
from .segmentation_main import segment_vegetation
from .config_loader import get_config, get_parameter, reload_config
from .interactive_visualization import InteractiveVisualization
from .static_visualization import StaticVisualization
from .json_exporter import VegetationClusterJSONExporter
from .spatial_bridging import SpatialBridging, BridgingParameters, apply_spatial_bridging_to_clusters

__all__ = [
    # Main API
    'segment_vegetation',
    
    # Core classes
    'VegetationSegmentationParameters',
    'STCube',
    'VegetationNDVIClusteringInitializer',
    
    # Spatial bridging
    'SpatialBridging',
    'BridgingParameters', 
    'apply_spatial_bridging_to_clusters',
    
    # Export functionality
    'VegetationClusterJSONExporter',
    
    # Configuration
    'get_config',
    'get_parameter',
    'reload_config',
]

# Add visualization classes if they imported successfully
if InteractiveVisualization is not None:
    __all__.append('InteractiveVisualization')
if StaticVisualization is not None:
    __all__.append('StaticVisualization')