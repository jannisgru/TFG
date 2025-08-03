"""
STCube Data Structures and Utilities for Vegetation Segmentation

Defines the STCube class representing a spatiotemporal vegetation cluster, including spatial, temporal, and NDVI profile properties. Also provides CubeCollection for efficient management and querying of multiple cubes, with spatial and temporal indexing support.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set
from collections import defaultdict
import rtree.index
import warnings
from ..config_loader import get_config

warnings.filterwarnings('ignore')


@dataclass
class STCube:
    """
    Represents a spatiotemporal cube.
    
    A spatiotemporal cube is a collection of pixels that are:
    - Spatially connected (2D footprint)
    - Temporally continuous (persistent over time)
    - Homogeneous in both space and time
    """
    id: int
    temporal_extent: Tuple[int, int]  # (start_time, end_time)
    pixels: Set[Tuple[int, int]]  # Set for O(1) lookups and automatic deduplication
    spectral_signature: np.ndarray  # Mean spectral values over time
    heterogeneity: float
    area: int
    perimeter: float
    compactness: float
    smoothness: float
    temporal_variance: float
    ndvi_profile: Optional[np.ndarray] = None
    _bbox: Optional[Tuple[int, int, int, int]] = field(default=None, init=False)  # Cached bbox
    
    def __post_init__(self):
        """Convert pixels to set if it's a list"""
        if isinstance(self.pixels, list):
            self.pixels = set(self.pixels)
        self.area = len(self.pixels)  # Ensure area matches pixel count
    
    @property
    def spatial_extent(self) -> Optional[Tuple[int, int, int, int]]:
        """Calculate spatial extent with caching"""
        if self._bbox is None:
            self._bbox = self._calculate_bbox()
        return self._bbox
    
    def _calculate_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """Calculate bounding box coordinates"""
        if not self.pixels:
            return None
        
        y_coords = [p[0] for p in self.pixels]
        x_coords = [p[1] for p in self.pixels]
        
        return (min(y_coords), max(y_coords), min(x_coords), max(x_coords))
    
    @property
    def duration(self) -> int:
        """Get temporal duration of the cube"""
        return self.temporal_extent[1] - self.temporal_extent[0] + 1
    
    def __hash__(self):
        """Make STCube hashable based on id for use in sets"""
        return hash(self.id)
    
    def __eq__(self, other):
        """Equality based on id for consistent behavior with hash"""
        if not isinstance(other, STCube):
            return False
        return self.id == other.id
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """Get bounding box coordinates (min_y, max_y, min_x, max_x)"""
        bbox = self.spatial_extent
        return bbox if bbox else (0, 0, 0, 0)
    
    def overlaps_spatially(self, other: 'STCube', margin: int = None) -> bool:
        """
        Check if this cube spatially overlaps with another cube using bounding boxes.
        """
        if margin is None:
            margin = get_config().spatial_margin
            
        bbox1 = self.get_bounding_box()
        bbox2 = other.get_bounding_box()
        
        if not bbox1 or not bbox2:
            return False
        
        min_y1, max_y1, min_x1, max_x1 = bbox1
        min_y2, max_y2, min_x2, max_x2 = bbox2
        
        # Expand first bbox by margin
        min_y1 -= margin
        max_y1 += margin
        min_x1 -= margin
        max_x1 += margin
        
        # Check for overlap
        return not (max_y1 < min_y2 or max_y2 < min_y1 or 
                   max_x1 < min_x2 or max_x2 < min_x1)
    
    def overlaps_spatially_exact(self, other: 'STCube') -> bool:
        """Check for exact pixel overlap (more expensive but precise)"""
        return bool(self.pixels.intersection(other.pixels))
    
    def overlaps_temporally(self, other: 'STCube', margin: int = None) -> bool:
        """Check if this cube temporally overlaps with another cube."""
        if margin is None:
            margin = get_config().temporal_margin
            
        start1, end1 = self.temporal_extent
        start2, end2 = other.temporal_extent
        
        # Expand temporal extents by margin
        start1 -= margin
        end1 += margin
        
        # Check for overlap
        return not (end1 < start2 or end2 < start1)
    
    def is_adjacent_to(self, other: 'STCube') -> bool:
        """Check if this cube is spatially or temporally adjacent to another cube."""
        # Check spatial adjacency (with 1-pixel margin)
        spatially_close = self.overlaps_spatially(other, margin=1)
        
        # Check temporal adjacency
        temporally_adjacent = (
            abs(self.temporal_extent[1] - other.temporal_extent[0]) <= 1 or
            abs(other.temporal_extent[1] - self.temporal_extent[0]) <= 1
        )
        
        return spatially_close and temporally_adjacent
    
    def merge_with(self, other: 'STCube', new_id: int) -> 'STCube':
        """
        Create a new cube by merging this cube with another (improved logic).
        """
        # Combine pixels (set automatically handles duplicates)
        combined_pixels = self.pixels.union(other.pixels)
        
        # Combine temporal extent
        combined_time_range = (
            min(self.temporal_extent[0], other.temporal_extent[0]),
            max(self.temporal_extent[1], other.temporal_extent[1])
        )
        
        # Compute area-weighted average for spectral signature
        total_area = len(combined_pixels)
        w1, w2 = len(self.pixels) / total_area, len(other.pixels) / total_area
        combined_signature = w1 * self.spectral_signature + w2 * other.spectral_signature
        
        # Combine NDVI profiles
        combined_ndvi_profile = None
        if self.ndvi_profile is not None and other.ndvi_profile is not None:
            combined_ndvi_profile = w1 * self.ndvi_profile + w2 * other.ndvi_profile
        elif self.ndvi_profile is not None:
            combined_ndvi_profile = self.ndvi_profile
        elif other.ndvi_profile is not None:
            combined_ndvi_profile = other.ndvi_profile
        
        # Calculate area-weighted averages for scalar properties
        combined_heterogeneity = w1 * self.heterogeneity + w2 * other.heterogeneity
        combined_temporal_variance = w1 * self.temporal_variance + w2 * other.temporal_variance
        combined_perimeter = self.perimeter + other.perimeter  # Approximation
        combined_compactness = w1 * self.compactness + w2 * other.compactness
        combined_smoothness = w1 * self.smoothness + w2 * other.smoothness
        
        # Create merged cube
        merged_cube = STCube(
            id=new_id,
            temporal_extent=combined_time_range,
            pixels=combined_pixels,
            spectral_signature=combined_signature,
            heterogeneity=combined_heterogeneity,
            area=total_area,
            perimeter=combined_perimeter,
            compactness=combined_compactness,
            smoothness=combined_smoothness,
            temporal_variance=combined_temporal_variance,
            ndvi_profile=combined_ndvi_profile
        )
        
        return merged_cube
    
    def invalidate_cache(self):
        """Invalidate cached computations when pixels change"""
        self._bbox = None
    
    def __repr__(self) -> str:
        return (f"STCube(id={self.id}, area={self.area}, "
                f"temporal_extent={self.temporal_extent}, "
                f"heterogeneity={self.heterogeneity:.4f})")


class CubeCollection:
    """
    Utility class for managing collections of STCubes.

    Uses spatial indexing for efficient neighbor queries.
    """
    
    def __init__(self, cubes: List[STCube]):
        self.cubes = cubes
        self._cube_dict = {cube.id: cube for cube in cubes}
        self._spatial_index = None
        self._temporal_index = None
        self._build_indices()
    
    def _build_indices(self):
        """Build spatial and temporal indices for efficient queries"""
        # Build spatial R-tree index
        self._spatial_index = rtree.index.Index()
        for cube in self.cubes:
            bbox = cube.get_bounding_box()
            if bbox != (0, 0, 0, 0):  # Valid bbox
                min_y, max_y, min_x, max_x = bbox
                # rtree expects (minx, miny, maxx, maxy)
                self._spatial_index.insert(cube.id, (min_x, min_y, max_x, max_y))
        
        # Build temporal index (simple dict grouping)
        self._temporal_index = defaultdict(list)
        for cube in self.cubes:
            start, end = cube.temporal_extent
            for t in range(start, end + 1):
                self._temporal_index[t].append(cube)
    
    def __len__(self) -> int:
        return len(self.cubes)
    
    def __iter__(self):
        return iter(self.cubes)
    
    def get_cube_by_id(self, cube_id: int) -> Optional[STCube]:
        """Get cube by ID"""
        return self._cube_dict.get(cube_id)
    
    def get_spatial_neighbors(self, cube: STCube, max_neighbors: int = None, 
                            margin: int = None) -> List[STCube]:
        """
        Find spatial neighbors using spatial index (much faster).
        """
        if max_neighbors is None:
            max_neighbors = get_config().max_neighbors
        if margin is None:
            margin = get_config().search_margin
        bbox = cube.get_bounding_box()
        if bbox == (0, 0, 0, 0):
            return []
        
        min_y, max_y, min_x, max_x = bbox
        
        # Expand search area by margin
        search_bbox = (min_x - margin, min_y - margin, 
                      max_x + margin, max_y + margin)
        
        # Query spatial index
        candidate_ids = list(self._spatial_index.intersection(search_bbox))
        
        # Filter out self and get actual cubes
        neighbors = []
        for cube_id in candidate_ids:
            if cube_id != cube.id:
                neighbor = self._cube_dict.get(cube_id)
                if neighbor and len(neighbors) < max_neighbors:
                    neighbors.append(neighbor)
        
        return neighbors
    
    def get_temporal_neighbors(self, cube: STCube, margin: int = None) -> List[STCube]:
        """Find temporal neighbors using temporal index."""
        if margin is None:
            margin = get_config().temporal_margin
            
        start, end = cube.temporal_extent
        neighbors = set()
        
        # Search in extended temporal range
        for t in range(start - margin, end + margin + 1):
            for temporal_cube in self._temporal_index.get(t, []):
                if (temporal_cube.id != cube.id and 
                    cube.overlaps_spatially(temporal_cube)):
                    neighbors.add(temporal_cube)
        
        return list(neighbors)
    
    def get_adjacent_cubes(self, cube: STCube) -> List[STCube]:
        """Find all adjacent cubes using indices."""
        # Get spatial candidates first
        config = get_config()
        spatial_candidates = self.get_spatial_neighbors(cube, max_neighbors=config.adjacency_search_neighbors, margin=1)
        
        # Filter for actual adjacency
        adjacent = []
        for candidate in spatial_candidates:
            if cube.is_adjacent_to(candidate):
                adjacent.append(candidate)
        
        return adjacent
    
    def remove_cubes(self, cube_ids: List[int]):
        """Remove cubes and rebuild indices"""
        cube_ids_set = set(cube_ids)
        self.cubes = [cube for cube in self.cubes if cube.id not in cube_ids_set]
        self._cube_dict = {cube.id: cube for cube in self.cubes}
        self._build_indices()  # Rebuild indices
    
    def add_cube(self, cube: STCube):
        """Add a new cube and update indices"""
        self.cubes.append(cube)
        self._cube_dict[cube.id] = cube
        
        # Update spatial index
        bbox = cube.get_bounding_box()
        if bbox != (0, 0, 0, 0):
            min_y, max_y, min_x, max_x = bbox
            self._spatial_index.insert(cube.id, (min_x, min_y, max_x, max_y))
        
        # Update temporal index
        start, end = cube.temporal_extent
        for t in range(start, end + 1):
            self._temporal_index[t].append(cube)
    
    def get_largest_cubes(self, n: int = 10) -> List[STCube]:
        """Get the n largest cubes by area"""
        if not self.cubes:
            return []
        
        areas = np.array([cube.area for cube in self.cubes])
        largest_indices = np.argpartition(areas, -min(n, len(areas)))[-n:]
        largest_indices = largest_indices[np.argsort(areas[largest_indices])[::-1]]
        
        return [self.cubes[i] for i in largest_indices]
    
    def get_stats(self) -> dict:
        """Get basic statistics about the cube collection (vectorized)"""
        if not self.cubes:
            return {}
        
        # Vectorized computations
        areas = np.array([cube.area for cube in self.cubes])
        durations = np.array([cube.duration for cube in self.cubes])
        heterogeneities = np.array([cube.heterogeneity for cube in self.cubes 
                                  if not np.isinf(cube.heterogeneity)])
        
        stats = {
            'n_cubes': len(self.cubes),
            'total_area': int(np.sum(areas)),
            'mean_area': float(np.mean(areas)),
            'std_area': float(np.std(areas)),
            'min_area': int(np.min(areas)),
            'max_area': int(np.max(areas)),
            'mean_duration': float(np.mean(durations)),
            'std_duration': float(np.std(durations)),
        }
        
        if len(heterogeneities) > 0:
            stats.update({
                'mean_heterogeneity': float(np.mean(heterogeneities)),
                'std_heterogeneity': float(np.std(heterogeneities)),
            })
        else:
            stats.update({
                'mean_heterogeneity': 0.0,
                'std_heterogeneity': 0.0,
            })
        
        return stats
    
    def get_cubes_in_region(self, bbox: Tuple[int, int, int, int], 
                           time_range: Optional[Tuple[int, int]] = None) -> List[STCube]:
        """Get all cubes that intersect with a spatial bounding box and optional time range"""
        min_y, max_y, min_x, max_x = bbox
        search_bbox = (min_x, min_y, max_x, max_y)
        
        candidate_ids = list(self._spatial_index.intersection(search_bbox))
        candidates = [self._cube_dict[cube_id] for cube_id in candidate_ids]
        
        if time_range:
            start_time, end_time = time_range
            candidates = [cube for cube in candidates 
                         if cube.temporal_extent[0] <= end_time and 
                            cube.temporal_extent[1] >= start_time]
        
        return candidates