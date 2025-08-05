"""
Visualization Modules for Vegetation Segmentation Results

This package provides static and interactive visualization capabilities
for the results of vegetation-focused ST-Cube segmentation.

Components:
- StaticVisualization: Publication-ready matplotlib visualizations (2D)
- InteractiveVisualization: Interactive 3D Plotly visualizations (3D)
- create_dual_trend_spatial_map: Static dual trend spatial visualization
- create_interactive_temporal_trend_map: Interactive temporal trend visualization
"""

from .visualization_2d import StaticVisualization
from .visualization_3d import InteractiveVisualization
from .common import create_dual_trend_spatial_map, create_interactive_temporal_trend_map

__all__ = [
    'StaticVisualization',
    'InteractiveVisualization',
    'create_dual_trend_spatial_map',
    'create_interactive_temporal_trend_map'
]
