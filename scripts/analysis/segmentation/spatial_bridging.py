"""
Spatial Bridging Module for Vegetation ST-Cube Segmentation

Implements graph-based spatial bridging to merge clusters/cubes that are connected by chains of similar pixels, 
even if separated by gaps. Uses NetworkX for graph construction and path finding, with configurable bridging criteria 
for NDVI similarity and spatial connectivity.
"""

import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from sklearn.neighbors import NearestNeighbors
from collections import deque
import networkx as nx
from dataclasses import dataclass
from loguru import logger

@dataclass
class BridgingParameters:
    """Parameters for spatial bridging algorithm"""
    bridge_similarity_tolerance: float = 0.1      # NDVI difference allowed for bridging pixels
    max_bridge_gap: int = 2                       # Maximum consecutive dissimilar pixels in bridge
    min_bridge_density: float = 0.7               # Minimum % of similar pixels in bridge path
    connectivity_radius: int = 3                  # Local neighborhood size for spatial graph
    max_bridge_length: int = 20                   # Maximum bridge path length to consider
    min_cluster_size_for_bridging: int = 5        # Only bridge clusters with this minimum size


class SpatialBridging:
    """
    Implements spatial bridging for cluster merging.
    
    The algorithm works in two phases:
    1. Build a spatial connectivity graph of pixels
    2. Find bridging paths between clusters and merge compatible ones
    """
    
    def __init__(self, params: BridgingParameters):
        self.params = params
        self.logger = logger
    
    def apply_spatial_bridging(self, 
                              cluster_labels: np.ndarray, 
                              pixel_coords: List[Tuple[int, int]], 
                              ndvi_profiles: np.ndarray) -> np.ndarray:
        """
        Apply spatial bridging to merge compatible clusters.
        
        Args:
            cluster_labels: Initial cluster labels for each pixel
            pixel_coords: List of (y, x) coordinates for each pixel
            ndvi_profiles: NDVI time series for each pixel
            
        Returns:
            Updated cluster labels after bridging
        """
        if len(cluster_labels) == 0:
            return cluster_labels
            
        self.logger.info("Starting spatial bridging analysis...")
        
        # Phase 1: Build spatial connectivity graph
        spatial_graph = self._build_spatial_graph(pixel_coords)
        
        # Phase 2: Analyze clusters and find bridging opportunities
        cluster_info = self._analyze_clusters(cluster_labels, pixel_coords, ndvi_profiles)
        
        # Phase 3: Find and apply bridges
        updated_labels = self._find_and_apply_bridges(
            cluster_labels, cluster_info, spatial_graph, pixel_coords, ndvi_profiles
        )
        
        n_original = len(np.unique(cluster_labels[cluster_labels >= 0]))
        n_final = len(np.unique(updated_labels[updated_labels >= 0]))
        logger.info(f"Spatial bridging: {n_original} -> {n_final} clusters")
        
        return updated_labels
    
    def _build_spatial_graph(self, pixel_coords: List[Tuple[int, int]]) -> nx.Graph:
        """Build spatial connectivity graph using k-nearest neighbors."""
        if not pixel_coords:
            return nx.Graph()
            
        coords_array = np.array(pixel_coords)
        
        # Use NearestNeighbors to find spatial connections
        nbrs = NearestNeighbors(
            n_neighbors=min(self.params.connectivity_radius * 4, len(pixel_coords)), 
            algorithm='kd_tree'
        )
        nbrs.fit(coords_array)
        
        # Build graph
        graph = nx.Graph()
        graph.add_nodes_from(range(len(pixel_coords)))
        
        for i, coord in enumerate(coords_array):
            distances, indices = nbrs.kneighbors([coord])
            
            for j, neighbor_idx in enumerate(indices[0]):
                if neighbor_idx != i:  # Don't connect to self
                    distance = distances[0][j]
                    if distance <= self.params.connectivity_radius:
                        graph.add_edge(i, neighbor_idx, weight=distance)
        
        return graph
    
    def _analyze_clusters(self, 
                         cluster_labels: np.ndarray, 
                         pixel_coords: List[Tuple[int, int]], 
                         ndvi_profiles: np.ndarray) -> Dict[int, Dict]:
        """Analyze cluster properties for bridging decisions."""
        cluster_info = {}
        
        for label in np.unique(cluster_labels):
            if label < 0:  # Skip noise points
                continue
                
            mask = cluster_labels == label
            cluster_pixels = np.where(mask)[0]
            
            if len(cluster_pixels) < self.params.min_cluster_size_for_bridging:
                continue
                
            cluster_coords = [pixel_coords[i] for i in cluster_pixels]
            cluster_ndvi = ndvi_profiles[cluster_pixels]
            
            cluster_info[label] = {
                'pixels': cluster_pixels,
                'coords': cluster_coords,
                'mean_ndvi': np.mean(cluster_ndvi, axis=0),
                'std_ndvi': np.std(cluster_ndvi, axis=0),
                'size': len(cluster_pixels)
            }
        
        return cluster_info
    
    def _find_and_apply_bridges(self, 
                               cluster_labels: np.ndarray, 
                               cluster_info: Dict[int, Dict], 
                               spatial_graph: nx.Graph, 
                               pixel_coords: List[Tuple[int, int]], 
                               ndvi_profiles: np.ndarray) -> np.ndarray:
        """Find bridging paths between clusters and merge compatible ones."""
        updated_labels = cluster_labels.copy()
        merge_pairs = []
        
        cluster_ids = list(cluster_info.keys())
        
        # Check all pairs of clusters for potential bridging
        for i, cluster_a in enumerate(cluster_ids):
            for cluster_b in cluster_ids[i+1:]:
                
                if self._clusters_are_compatible(cluster_info[cluster_a], cluster_info[cluster_b]):
                    bridge_path = self._find_bridge_path(
                        cluster_info[cluster_a], cluster_info[cluster_b], 
                        spatial_graph, ndvi_profiles
                    )
                    
                    if bridge_path and self._validate_bridge_quality(bridge_path, ndvi_profiles, cluster_info[cluster_a], cluster_info[cluster_b]):
                        merge_pairs.append((cluster_a, cluster_b))
        
        # Apply merges (merge smaller cluster into larger one)
        for cluster_a, cluster_b in merge_pairs:
            size_a = cluster_info[cluster_a]['size']
            size_b = cluster_info[cluster_b]['size']
            
            if size_a >= size_b:
                target, source = cluster_a, cluster_b
            else:
                target, source = cluster_b, cluster_a
            
            # Merge source into target
            updated_labels[updated_labels == source] = target
        
        return updated_labels
    
    def _clusters_are_compatible(self, cluster_a: Dict, cluster_b: Dict) -> bool:
        """Check if two clusters are compatible for bridging based on NDVI similarity."""
        mean_diff = np.mean(np.abs(cluster_a['mean_ndvi'] - cluster_b['mean_ndvi']))
        return mean_diff <= self.params.bridge_similarity_tolerance
    
    def _find_bridge_path(self, 
                         cluster_a: Dict, 
                         cluster_b: Dict, 
                         spatial_graph: nx.Graph, 
                         ndvi_profiles: np.ndarray) -> Optional[List[int]]:
        """Find shortest path between clusters through spatial graph."""
        
        # Find boundary pixels (pixels with neighbors in the other cluster's vicinity)
        boundary_a = self._find_boundary_pixels(cluster_a['pixels'], spatial_graph)
        boundary_b = self._find_boundary_pixels(cluster_b['pixels'], spatial_graph)
        
        shortest_path = None
        min_length = float('inf')
        
        # Try to find path between boundary pixels
        for pixel_a in boundary_a[:5]:  # Limit to avoid too much computation
            for pixel_b in boundary_b[:5]:
                try:
                    if nx.has_path(spatial_graph, pixel_a, pixel_b):
                        path = nx.shortest_path(spatial_graph, pixel_a, pixel_b)
                        if len(path) <= self.params.max_bridge_length and len(path) < min_length:
                            shortest_path = path
                            min_length = len(path)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
        
        return shortest_path
    
    def _find_boundary_pixels(self, cluster_pixels: List[int], spatial_graph: nx.Graph) -> List[int]:
        """Find boundary pixels of a cluster (pixels with external neighbors)."""
        boundary = []
        
        for pixel in cluster_pixels:
            if pixel in spatial_graph:
                neighbors = list(spatial_graph.neighbors(pixel))
                # Check if any neighbor is outside the cluster
                external_neighbors = [n for n in neighbors if n not in cluster_pixels]
                if external_neighbors:
                    boundary.append(pixel)
        
        return boundary
    
    def _validate_bridge_quality(self, 
                                bridge_path: List[int], 
                                ndvi_profiles: np.ndarray, 
                                cluster_a: Dict, 
                                cluster_b: Dict) -> bool:
        """Validate if a bridge path has sufficient quality for merging."""
        if len(bridge_path) <= 2:  # Direct connection
            return True
            
        # Calculate target NDVI (interpolation between cluster means)
        path_length = len(bridge_path)
        target_ndvis = []
        
        for i, pixel_idx in enumerate(bridge_path):
            # Interpolate between cluster means based on position in path
            weight_a = 1.0 - (i / (path_length - 1))
            weight_b = i / (path_length - 1)
            target_ndvi = weight_a * cluster_a['mean_ndvi'] + weight_b * cluster_b['mean_ndvi']
            target_ndvis.append(target_ndvi)
        
        # Check bridge quality
        similar_pixels = 0
        consecutive_gaps = 0
        max_consecutive_gaps = 0
        
        for i, pixel_idx in enumerate(bridge_path):
            pixel_ndvi = ndvi_profiles[pixel_idx]
            target_ndvi = target_ndvis[i]
            
            # Check similarity
            ndvi_diff = np.mean(np.abs(pixel_ndvi - target_ndvi))
            
            if ndvi_diff <= self.params.bridge_similarity_tolerance:
                similar_pixels += 1
                consecutive_gaps = 0
            else:
                consecutive_gaps += 1
                max_consecutive_gaps = max(max_consecutive_gaps, consecutive_gaps)
        
        # Validate quality metrics
        bridge_density = similar_pixels / len(bridge_path)
        
        quality_checks = [
            bridge_density >= self.params.min_bridge_density,
            max_consecutive_gaps <= self.params.max_bridge_gap
        ]
        
        return all(quality_checks)


def apply_spatial_bridging_to_clusters(cluster_labels: np.ndarray, 
                                     pixel_coords: List[Tuple[int, int]], 
                                     ndvi_profiles: np.ndarray,
                                     bridging_params: Optional[BridgingParameters] = None) -> np.ndarray:
    """
    Convenience function to apply spatial bridging to existing clusters.
    
    Args:
        cluster_labels: Initial cluster labels
        pixel_coords: Pixel coordinates
        ndvi_profiles: NDVI time series data
        bridging_params: Bridging parameters (uses defaults if None)
        
    Returns:
        Updated cluster labels after spatial bridging
    """
    if bridging_params is None:
        bridging_params = BridgingParameters()
    
    bridging = SpatialBridging(bridging_params)
    return bridging.apply_spatial_bridging(cluster_labels, pixel_coords, ndvi_profiles)
