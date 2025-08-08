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