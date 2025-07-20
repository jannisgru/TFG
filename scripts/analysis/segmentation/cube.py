"""
ST-Cube data structure and related utilities.

This module defines the core STCube class that represents a spatiotemporal cube
in the segmentation algorithm.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class STCube:
    """
    Represents a spatiotemporal cube (memory-efficient version).
    
    A spatiotemporal cube is a collection of pixels that are:
    - Spatially connected (2D footprint)
    - Temporally continuous (persistent over time)
    - Homogeneous in both space and time
    """
    id: int
    temporal_extent: Tuple[int, int]  # (start_time, end_time)
    pixels: List[Tuple[int, int]]  # List of (y, x) coordinates
    spectral_signature: np.ndarray  # Mean spectral values over time
    heterogeneity: float
    area: int
    perimeter: float
    compactness: float
    smoothness: float
    temporal_variance: float
    ndvi_profile: Optional[np.ndarray] = None  # NDVI time series profile
    
    @property
    def spatial_extent(self) -> Optional[Tuple[int, int, int, int]]:
        """Calculate spatial extent on demand to save memory"""
        if not self.pixels:
            return None
        min_y = min(p[0] for p in self.pixels)
        max_y = max(p[0] for p in self.pixels)
        min_x = min(p[1] for p in self.pixels)
        max_x = max(p[1] for p in self.pixels)
        return (min_y, max_y, min_x, max_x)
    
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
        if not self.pixels:
            return (0, 0, 0, 0)
        
        y_coords = [p[0] for p in self.pixels]
        x_coords = [p[1] for p in self.pixels]
        
        return (min(y_coords), max(y_coords), min(x_coords), max(x_coords))
    
    def overlaps_spatially(self, other: 'STCube', margin: int = 1) -> bool:
        """
        Check if this cube spatially overlaps with another cube.
        
        Args:
            other: Another STCube to check overlap with
            margin: Spatial margin for overlap detection
            
        Returns:
            True if cubes overlap spatially
        """
        bbox1 = self.get_bounding_box()
        bbox2 = other.get_bounding_box()
        
        # Expand bounding boxes by margin
        min_y1, max_y1, min_x1, max_x1 = bbox1
        min_y2, max_y2, min_x2, max_x2 = bbox2
        
        min_y1 -= margin
        max_y1 += margin
        min_x1 -= margin
        max_x1 += margin
        
        # Check for overlap
        return not (max_y1 < min_y2 or max_y2 < min_y1 or 
                   max_x1 < min_x2 or max_x2 < min_x1)
    
    def overlaps_temporally(self, other: 'STCube', margin: int = 0) -> bool:
        """
        Check if this cube temporally overlaps with another cube.
        
        Args:
            other: Another STCube to check overlap with
            margin: Temporal margin for overlap detection
            
        Returns:
            True if cubes overlap temporally
        """
        start1, end1 = self.temporal_extent
        start2, end2 = other.temporal_extent
        
        # Expand temporal extents by margin
        start1 -= margin
        end1 += margin
        
        # Check for overlap
        return not (end1 < start2 or end2 < start1)
    
    def is_adjacent_to(self, other: 'STCube') -> bool:
        """
        Check if this cube is spatially or temporally adjacent to another cube.
        
        Args:
            other: Another STCube to check adjacency with
            
        Returns:
            True if cubes are adjacent
        """
        # Check spatial adjacency (with 1-pixel margin)
        spatially_adjacent = self.overlaps_spatially(other, margin=1)
        
        # Check temporal adjacency
        temporally_adjacent = (
            abs(self.temporal_extent[1] - other.temporal_extent[0]) <= 1 or
            abs(other.temporal_extent[1] - self.temporal_extent[0]) <= 1
        )
        
        return spatially_adjacent and temporally_adjacent
    
    def merge_with(self, other: 'STCube', new_id: int) -> 'STCube':
        """
        Create a new cube by merging this cube with another.
        
        Args:
            other: Another STCube to merge with
            new_id: ID for the new merged cube
            
        Returns:
            New STCube representing the merged result
        """
        # Combine properties
        combined_pixels = self.pixels + other.pixels
        combined_time_range = (
            min(self.temporal_extent[0], other.temporal_extent[0]),
            max(self.temporal_extent[1], other.temporal_extent[1])
        )
        
        # Combine spectral signatures
        combined_signature = np.concatenate([self.spectral_signature, other.spectral_signature])
        
        # Combine NDVI profiles if available
        combined_ndvi_profile = None
        if self.ndvi_profile is not None and other.ndvi_profile is not None:
            combined_ndvi_profile = np.concatenate([self.ndvi_profile, other.ndvi_profile])
        elif self.ndvi_profile is not None:
            combined_ndvi_profile = self.ndvi_profile
        elif other.ndvi_profile is not None:
            combined_ndvi_profile = other.ndvi_profile
        
        # Calculate weighted averages for scalar properties
        total_area = self.area + other.area
        combined_heterogeneity = (
            (self.heterogeneity * self.area + other.heterogeneity * other.area) / total_area
        )
        combined_temporal_variance = (
            (self.temporal_variance * self.area + other.temporal_variance * other.area) / total_area
        )
        
        # Create merged cube
        merged_cube = STCube(
            id=new_id,
            temporal_extent=combined_time_range,
            pixels=combined_pixels,
            spectral_signature=combined_signature,
            heterogeneity=combined_heterogeneity,
            area=total_area,
            perimeter=0.0,  # Will be calculated if needed
            compactness=0.0,  # Will be calculated if needed
            smoothness=0.0,  # Will be calculated if needed
            temporal_variance=combined_temporal_variance,
            ndvi_profile=combined_ndvi_profile
        )
        
        return merged_cube
    
    def __repr__(self) -> str:
        return (f"STCube(id={self.id}, area={self.area}, "
                f"temporal_extent={self.temporal_extent}, "
                f"heterogeneity={self.heterogeneity:.4f})")


class CubeCollection:
    """
    Utility class for managing collections of STCubes.
    
    Provides efficient operations for finding neighbors, spatial queries, etc.
    """
    
    def __init__(self, cubes: List[STCube]):
        self.cubes = cubes
        self._cube_dict = {cube.id: cube for cube in cubes}
    
    def __len__(self) -> int:
        return len(self.cubes)
    
    def __iter__(self):
        return iter(self.cubes)
    
    def get_cube_by_id(self, cube_id: int) -> Optional[STCube]:
        """Get cube by ID"""
        return self._cube_dict.get(cube_id)
    
    def get_spatial_neighbors(self, cube: STCube, max_neighbors: int = 10) -> List[STCube]:
        """
        Find spatial neighbors of a cube.
        
        Args:
            cube: The cube to find neighbors for
            max_neighbors: Maximum number of neighbors to return
            
        Returns:
            List of neighboring cubes
        """
        neighbors = []
        
        for other_cube in self.cubes:
            if other_cube.id != cube.id and cube.overlaps_spatially(other_cube, margin=3):
                neighbors.append(other_cube)
                
                if len(neighbors) >= max_neighbors:
                    break
        
        return neighbors
    
    def get_temporal_neighbors(self, cube: STCube) -> List[STCube]:
        """
        Find temporal neighbors of a cube.
        
        Args:
            cube: The cube to find neighbors for
            
        Returns:
            List of temporally neighboring cubes
        """
        neighbors = []
        
        for other_cube in self.cubes:
            if (other_cube.id != cube.id and 
                cube.overlaps_temporally(other_cube, margin=1) and
                cube.overlaps_spatially(other_cube)):
                neighbors.append(other_cube)
        
        return neighbors
    
    def get_adjacent_cubes(self, cube: STCube) -> List[STCube]:
        """
        Find all adjacent cubes (spatially and temporally).
        
        Args:
            cube: The cube to find adjacent cubes for
            
        Returns:
            List of adjacent cubes
        """
        adjacent = []
        
        for other_cube in self.cubes:
            if other_cube.id != cube.id and cube.is_adjacent_to(other_cube):
                adjacent.append(other_cube)
        
        return adjacent
    
    def remove_cubes(self, cube_ids: List[int]):
        """Remove cubes with specified IDs"""
        self.cubes = [cube for cube in self.cubes if cube.id not in cube_ids]
        self._cube_dict = {cube.id: cube for cube in self.cubes}
    
    def add_cube(self, cube: STCube):
        """Add a new cube to the collection"""
        self.cubes.append(cube)
        self._cube_dict[cube.id] = cube
    
    def get_largest_cubes(self, n: int = 10) -> List[STCube]:
        """Get the n largest cubes by area"""
        return sorted(self.cubes, key=lambda c: c.area, reverse=True)[:n]
    
    def get_stats(self) -> dict:
        """Get basic statistics about the cube collection"""
        if not self.cubes:
            return {}
        
        areas = [cube.area for cube in self.cubes]
        durations = [cube.duration for cube in self.cubes]
        heterogeneities = [cube.heterogeneity for cube in self.cubes 
                          if not np.isinf(cube.heterogeneity)]
        
        return {
            'n_cubes': len(self.cubes),
            'total_area': sum(areas),
            'mean_area': np.mean(areas),
            'std_area': np.std(areas),
            'min_area': min(areas),
            'max_area': max(areas),
            'mean_duration': np.mean(durations),
            'std_duration': np.std(durations),
            'mean_heterogeneity': np.mean(heterogeneities) if heterogeneities else 0,
            'std_heterogeneity': np.std(heterogeneities) if heterogeneities else 0,
        }
