"""
Core Data Structures for Vegetation Segmentation

This package contains the fundamental data structures and parameter definitions
for vegetation-focused ST-Cube segmentation.

Components:
- VegetationSegmentationParameters: Core parameter dataclass
- STCube: Spatiotemporal cube representation
- CubeCollection: Efficient cube management utilities
"""

from .base import VegetationSegmentationParameters
from .cube import STCube, CubeCollection

__all__ = [
    'VegetationSegmentationParameters',
    'STCube',
    'CubeCollection'
]
