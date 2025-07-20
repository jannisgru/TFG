"""
Core parameters for vegetation-focused ST-Cube segmentation.

This module defines the essential parameters for NDVI-based 
spatiotemporal cube segmentation focusing on vegetation areas.
"""

import numpy as np
import xarray as xr
from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .cube import STCube


@dataclass
class VegetationSegmentationParameters:
    """Parameters for vegetation-focused ST-Cube segmentation"""
    sh_threshold: float = 0.06          # Spatial heterogeneity threshold
    th_threshold: float = 0.03          # Temporal heterogeneity threshold
    w_compactness: float = 0.5          # Weight for compactness
    min_cube_size: int = 20             # Minimum pixels per cube
    max_iterations: int = 20            # Maximum merge iterations
    max_spatial_distance: int = 10      # Maximum spatial distance for clustering
    min_vegetation_ndvi: float = 0.4    # Minimum NDVI for vegetation
    
    def __post_init__(self):
        if self.min_cube_size < 1:
            self.min_cube_size = 1
        if self.max_iterations < 1:
            self.max_iterations = 1
        if self.max_spatial_distance < 1:
            self.max_spatial_distance = 1
        if self.min_vegetation_ndvi < 0:
            self.min_vegetation_ndvi = 0.0
        if self.min_vegetation_ndvi > 1:
            self.min_vegetation_ndvi = 1.0
