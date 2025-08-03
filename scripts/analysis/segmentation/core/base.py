"""
Parameter Definitions for Vegetation-Focused ST-Cube Segmentation

This module defines the dataclass for all core segmentation parameters used in NDVI-based spatiotemporal cube segmentation, including defaults and config integration. It is the central place for parameter validation and loading from config.
"""

from dataclasses import dataclass
from typing import Optional
import warnings
from ..config_loader import get_config

warnings.filterwarnings('ignore')


@dataclass
class VegetationSegmentationParameters:
    """Parameters for vegetation-focused ST-Cube segmentation"""
    min_cube_size: int = None
    max_spatial_distance: int = None
    min_vegetation_ndvi: float = None
    n_clusters: Optional[int] = None
    ndvi_variance_threshold: float = None
    temporal_weight: float = None
    ndvi_trend_filter: Optional[str] = None  # 'greening', 'browning', or None
    
    def __post_init__(self):
        # Load config and set defaults if not provided
        config = get_config()
        
        if self.min_cube_size is None:
            self.min_cube_size = config.min_cube_size
        if self.max_spatial_distance is None:
            self.max_spatial_distance = config.max_spatial_distance
        if self.min_vegetation_ndvi is None:
            self.min_vegetation_ndvi = config.min_vegetation_ndvi
        if self.n_clusters is None:
            self.n_clusters = config.n_clusters
        if self.ndvi_variance_threshold is None:
            self.ndvi_variance_threshold = config.ndvi_variance_threshold
        if self.temporal_weight is None:
            self.temporal_weight = config.temporal_weight
        if self.ndvi_trend_filter is None:
            self.ndvi_trend_filter = config.ndvi_trend_filter
            
        # Validation
        if self.min_cube_size < 1:
            self.min_cube_size = 1
        if self.max_spatial_distance < 1:
            self.max_spatial_distance = 1
        if self.min_vegetation_ndvi < 0:
            self.min_vegetation_ndvi = 0.0
        if self.min_vegetation_ndvi > 1:
            self.min_vegetation_ndvi = 1.0
        if self.n_clusters is not None and self.n_clusters < 1:
            self.n_clusters = None
        if self.ndvi_trend_filter is not None and self.ndvi_trend_filter not in ['greening', 'browning']:
            self.ndvi_trend_filter = None
