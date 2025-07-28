"""
Core parameters for vegetation-focused ST-Cube segmentation.

This module defines the essential parameters for NDVI-based 
spatiotemporal cube segmentation focusing on vegetation areas.
"""

# ==== CONFIGURABLE PARAMETERS ====
DEFAULT_SH_THRESHOLD = 0.06          # Spatial heterogeneity threshold
DEFAULT_TH_THRESHOLD = 0.03          # Temporal heterogeneity threshold
DEFAULT_W_COMPACTNESS = 0.5          # Weight for compactness
DEFAULT_MIN_CUBE_SIZE = 20           # Minimum pixels per cube
DEFAULT_MAX_ITERATIONS = 20          # Maximum merge iterations
DEFAULT_MAX_SPATIAL_DISTANCE = 10    # Maximum spatial distance for clustering
DEFAULT_MIN_VEGETATION_NDVI = 0.4    # Minimum NDVI for vegetation
DEFAULT_N_CLUSTERS = None            # Maximum number of clusters to analyze (None = unlimited)
# ================================

from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class VegetationSegmentationParameters:
    """Parameters for vegetation-focused ST-Cube segmentation"""
    sh_threshold: float = DEFAULT_SH_THRESHOLD          # Spatial heterogeneity threshold
    th_threshold: float = DEFAULT_TH_THRESHOLD          # Temporal heterogeneity threshold
    w_compactness: float = DEFAULT_W_COMPACTNESS        # Weight for compactness
    min_cube_size: int = DEFAULT_MIN_CUBE_SIZE          # Minimum pixels per cube
    max_iterations: int = DEFAULT_MAX_ITERATIONS        # Maximum merge iterations
    max_spatial_distance: int = DEFAULT_MAX_SPATIAL_DISTANCE  # Maximum spatial distance for clustering
    min_vegetation_ndvi: float = DEFAULT_MIN_VEGETATION_NDVI  # Minimum NDVI for vegetation
    n_clusters: int = DEFAULT_N_CLUSTERS                # Maximum number of clusters to analyze (None = unlimited)
    
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
        if self.n_clusters is not None and self.n_clusters < 1:
            self.n_clusters = None
