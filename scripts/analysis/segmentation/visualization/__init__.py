"""
Visualization Modules for Vegetation Segmentation Results

This package provides static and interactive visualization capabilities
for the results of vegetation-focused ST-Cube segmentation.

Components:
- StaticVisualization: Publication-ready matplotlib visualizations (2D)
- InteractiveVisualization: Interactive 3D Plotly visualizations (3D)
"""

from .visualization_2d import StaticVisualization
from .visualization_3d import InteractiveVisualization

__all__ = [
    'StaticVisualization',
    'InteractiveVisualization'
]
