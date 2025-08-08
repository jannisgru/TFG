"""
Vegetation-focused ST-Cube Segmentation Package

A streamlined implementation of spatiotemporal trace segmentation focused specifically
on vegetation analysis using NDVI clustering with local spatial constraints.

Package Structure:
- visualization/: Static and interactive visualization modules
- Main modules: segmentation_main, config_loader, json_exporter

Main Function:
- segment_vegetation: Run vegetation-focused NDVI clustering segmentation

Features:
- NDVI-based clustering with spatial constraints
- Temporal trend filtering (greening/browning NDVI)
- Interactive and static visualizations
- JSON export for further analysis

Example Usage:
    # Vegetation NDVI clustering segmentation with config
    from segmentation import segment_vegetation, get_config
    
    # Use default config values
    traces = segment_vegetation()
    
    # Or customize specific parameters
    from segmentation.segmentation_main import VegetationSegmentationParameters
    params = VegetationSegmentationParameters(
        max_spatial_distance=10,
        min_vegetation_ndvi=0.4,
        min_cluster_size=20
    )
    
    traces = segment_vegetation(parameters=params)
    
    # Filter for only decreasing NDVI trends (e.g., deforestation)
    params = VegetationSegmentationParameters(
        max_spatial_distance=10,
        min_vegetation_ndvi=0.4,
        min_cluster_size=20,
        ndvi_trend_filter='decreasing'
    )
    
    traces = segment_vegetation(parameters=params)
"""
from .visualization import StaticVisualization, InteractiveVisualization, CommonVisualization
from .segmentation_main import segment_vegetation, VegetationSegmentationParameters
from .config_loader import get_config, get_parameter, reload_config
from .json_exporter import VegetationClusterJSONExporter

__all__ = [
    # Main API
    'segment_vegetation',
    
    # Core classes
    'VegetationSegmentationParameters',
    
    # Export functionality
    'VegetationClusterJSONExporter',
    
    # Configuration
    'get_config',
    'get_parameter',
    'reload_config',
    
    # Visualization
    'StaticVisualization',
    'InteractiveVisualization',
    'CommonVisualization'
]