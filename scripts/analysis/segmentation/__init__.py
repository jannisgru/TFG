"""
Vegetation-focused ST-Cube Segmentation Package

A streamlined implementation of spatiotemporal cube segmentation focused specifically
on vegetation analysis using NDVI clustering with local spatial constraints.

Package Structure:
- core/: Core data structures (VegetationSegmentationParameters, STCube, CubeCollection)
- visualization/: Static and interactive visualization modules
- initializers/: Clustering initialization strategies
- Main modules: segmentation_main, spatial_bridging, config_loader, json_exporter

Main Function:
- segment_vegetation: Run vegetation-focused NDVI clustering segmentation

Features:
- NDVI-based clustering with spatial constraints
- Temporal trend filtering (increasing/decreasing NDVI)
- Spatial bridging for cluster connectivity
- Interactive and static visualizations
- JSON export for further analysis

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
    
    # Filter for only decreasing NDVI trends (e.g., deforestation)
    params = VegetationSegmentationParameters(
        max_spatial_distance=10,
        min_vegetation_ndvi=0.4,
        min_cube_size=20,
        ndvi_trend_filter='decreasing'
    )
    
    cubes = segment_vegetation(parameters=params)
"""
from .core import VegetationSegmentationParameters, STCube
from .visualization import StaticVisualization, InteractiveVisualization
from .initializers import VegetationNDVIClusteringInitializer
from .segmentation_main import segment_vegetation
from .config_loader import get_config, get_parameter, reload_config
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
    
    # Visualization
    'StaticVisualization',
    'InteractiveVisualization'
]